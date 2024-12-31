import os
import argparse
import builtins
import time
import numpy as np

import torch
import json
import torch.nn.functional as F
from torch import multiprocessing as mp
import torch.distributed as dist

import utils
from model import AVENet
from DatasetLoader_origin import GetAudioVideoDataset
from tqdm import tqdm

# from datasets import get_train_dataset, get_test_dataset
from torch.utils.tensorboard import SummaryWriter



import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    
    # Model and experiment configurations
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--experiment_name', type=str, default='hdvsl_vggss', help='Experiment name for checkpointing and logging')

    # Dataset paths and configurations
    parser.add_argument('--trainset', type=str, default='vggss', help='Name of the training dataset (e.g., flickr, vggss)')
    parser.add_argument('--testset', type=str, default='vggss', help='Name of the test dataset (e.g., flickr, vggss)')
    parser.add_argument('--image_size',default=224,type=int,help='Height and width of inputs')

    # Model hyperparameters as used in AVENet class
    parser.add_argument('--epsilon', type=float, default=0.65, help='Threshold for positive cases in similarity calculation')
    parser.add_argument('--epsilon2', type=float, default=0.4, help='Threshold for negative cases in similarity calculation')
    # epsilon이랑 epsilon2 필요한지 아직 잘 모르겠음. (positive, negative 하지 말라는 것 같았음)

    parser.add_argument('--tri_map', action='store_true', help='Use tri-map for additional negative cases')
    parser.set_defaults(tri_map=True)
    parser.add_argument('--Neg', action='store_true', help='Include negative samples in similarity calculation')
    parser.set_defaults(Neg=True)
    parser.add_argument('--random_threshold', type=float, default=0.03, help='Threshold for random sampling (if used)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--init_lr', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed for reproducibility')

    # Distributed training parameters
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id to use')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of nodes for distributed training')
    parser.add_argument('--rank', type=int, default=0, help='Node rank for distributed training')
    parser.add_argument('--node', type=str, default='localhost', help='Node hostname')
    parser.add_argument('--port', type=int, default=12345, help='Port for distributed training communication')
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345', help='URL for distributed training')
    parser.add_argument('--multiprocessing_distributed', action='store_true', help='Use multi-processing distributed training')
    # parser.add_argument('--loss', type=str, default='baseline', help='Choosing loss function')

    # tensorboard 관련 parser
    parser.add_argument('--exp_dir', type=str, default="", help='Should be "./logs/exp#" form')
    parser.add_argument('--resume', dest="resume", action="store_true", help='Resume training from last checkpoint if True')


    return parser.parse_args()


def main(args):
    mp.set_start_method('spawn') # 다중 프로세스를 실행할 때 spawn 방식을 사용하여 각 프로세스를 독립적으로 시작.
    args.dist_url = f'tcp://{args.node}:{args.port}' # 분산 학습에서 사용할 URL을 생성
    print('Using url {}'.format(args.dist_url))

    ngpus_per_node = torch.cuda.device_count() # 현재 노드에서 사용할 수 있는 GPU 개수를 파악
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))

    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        
    # Setup distributed environment
    if args.multiprocessing_distributed:
        # 여기 안들어옴
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            # global rank를 설정 -> 모든 프로세스에서 고유한 rank를 갖도록 함
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier() # 모든 프로세스가 이 지점에 도달할 때까지 대기

    # Create model dir
    model_dir = os.path.join(args.model_dir, args.experiment_name) # ./checkpoints/hdvsl_vggss
    os.makedirs(model_dir, exist_ok=True)
    utils.save_json(vars(args), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)
    # sort_keys = True : key를 정렬하여 저장
    # save_pretty = True : json 파일을 보기 좋게 저장

    # Create model
    model = AVENet(args)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    print(model)

    # Optimizer
    optimizer, scheduler = utils.build_optimizer_and_scheduler_adam(model, args)

    # Resume if possible

    start_epoch, best_recall_at_10 = 0, 0.  # 초기화 단계로 처음에 변수를 0으로 설정
    # if os.path.exists(os.path.join(model_dir, 'latest.pth')): # model_dir = './checkpoints/hdvsl_vggss'
    #     ckp = torch.load(os.path.join(model_dir, 'latest.pth'), map_location='cpu') # latest.pth 파일을 로드하여 ckp에 저장
    #     start_epoch = ckp.get('epoch', 0)
    #     best_recall_at_10 = ckp.get('best_recall_at_10', 0.0)  # 로드한 checkpoint file에서 epoch, best_recall_at_10 값을 가져와 할당
    #     model.load_state_dict(ckp['model'])
    #     optimizer.load_state_dict(ckp['optimizer'])
    #     print(f'loaded from {os.path.join(model_dir, "latest.pth")}')


    # Dataloaders
    traindataset = GetAudioVideoDataset(args, mode = 'train') # train dataset 생성
    train_sampler = None
    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset) # train_sampler 초기화
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size= 16, shuffle = False, num_workers = 1, drop_last = True)
    
    testdataset = GetAudioVideoDataset(args,  mode='test') # test dataset 생성
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=128, shuffle=False, num_workers = 1)

    print("Loaded dataloader.")

    # =============================================================== #
    # Training loop
    recall_at_10 = validate(test_loader, model, args)
    print(f'Recall@10 (epoch {start_epoch}): {recall_at_10}')
    print(f'best Recall@10: {best_recall_at_10}')


    exp_dir = args.exp_dir
    exp_num = exp_dir.split('/')[-1]
    log_dir = f"./logs/{exp_num}"


    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # SummaryWriter를 한 번만 생성
    writer = SummaryWriter(log_dir)

    best_recall_at_10 = 0.0
    model_dir = args.model_dir

    saved_best_pth_path = os.path.join(model_dir, 'best.pth')
    if not os.path.exists(saved_best_pth_path):
        saved_best_recall_at_10 = 0.0
    else:
        ckp = torch.load(saved_best_pth_path)
        saved_best_recall_at_10 = ckp['best_recall_at_10']
    

    for epoch in range(start_epoch, args.epochs):
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train(train_loader, model, optimizer, epoch, args, writer)

        # Evaluate
        recall_at_10 = validate(test_loader, model, args)

        writer.add_scalar('Validation/Recall@10', recall_at_10, epoch)
        
        print(f'Recall@10 (epoch {epoch+1}): {recall_at_10}')
        print(f'best Recall@10: {best_recall_at_10}')


        # Checkpoint
        # if args.rank == 0: # 분산 학습을 사용할 때, rank가 0인 프로세스만 checkpoint를 저장
        #     ckp = {'model': model.state_dict(),
        #            'optimizer': optimizer.state_dict(),
        #            'epoch': epoch+1,
        #            'best_recall_at_10': best_recall_at_10}
        #     torch.save(ckp, os.path.join(model_dir, 'latest.pth'))
        #     print(f"Model saved to {model_dir}")

        # if recall_at_10 >= best_recall_at_10:
        #     best_recall_at_10 = recall_at_10
        #     if args.rank == 0:
        #         torch.save(ckp, os.path.join(model_dir, 'best.pth'))

        if recall_at_10 >= saved_best_recall_at_10:
            saved_best_recall_at_10 = recall_at_10
            if args.rank == 0:
                ckp = {'model': model.state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'epoch': epoch+1,
                       'best_recall_at_10': saved_best_recall_at_10}
                torch.save(ckp, saved_best_pth_path)
                print(f"New best model saved to {model_dir}")     


def load_labels(mode):
    if mode == 'train':
        label_path = "/mnt/scratch/users/individuals/VGGsound_individual/metadata/train_a_third.json"
    else:
        label_path = "/mnt/scratch/users/individuals/VGGsound_individual/metadata/test.json"

    with open(label_path, 'r') as f:
        full_data = json.load(f)  # JSON 파일 전체를 로드
        data = full_data["data"]  # "data" 키를 통해 실제 데이터에 접근

    labels = {}
    for item in data:
        labels[item['video_id']] = item['labels']
    
    return labels

# 최종 느낌의 train 이었음
def train(train_loader, model, optimizer, epoch, args, writer):
    print("train 들어옴")
    model.train() # 모델을 훈련 모드로 전환
    batch_time = AverageMeter('Time', ':6.3f') # 소수점 이하 3자리까지, 총 6자리
    data_time = AverageMeter('Data', ':6.3f') # 소수점 이하 3자리까지, 총 6자리
    loss_mtr = AverageMeter('Loss', ':.3f')
    recall_at_10_mtr = AverageMeter('Recall@10', ':.3f')


    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_mtr],
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    global_step = 0

    if args.resume:
        # 진행 상태 복원
        exp_dir = args.exp_dir
        progress_pkl = f"{exp_dir}/progress.pkl"
        progress_data, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
        print("\nResume training from:")
        print("  epoch = %s" % epoch)
        print("  global_step = %s" % global_step)
        print("  best_epoch = %s" % best_epoch)
        print("  best_acc = %.4f" % best_acc)


    for i, (image, spec, _, _, _) in tqdm(enumerate(train_loader), desc="Train Embedding Extraction", total=len(train_loader)):
        data_time.update(time.time() - end)

        # 데이터 GPU로 이동
        spec = spec.cuda()
        image = image.cuda()

        # 모델을 사용하여 임베딩 추출
        image_emb, audio_emb = model.extract_features(image, spec)
        # 각각 사이즈 (B, C)를 가짐

        # 유사도 행렬을 배치 단위로 계산
        similarity_matrix = torch.einsum("bc, ac -> ba", image_emb, audio_emb)
        similarity_matrix = similarity_matrix/0.07
        #similarity_matrix : (B, B) 크기의 유사도 행렬
        
        # Cross-Entropy Loss 계산
        # 이 부분 체크해보기
        labels = torch.arange(similarity_matrix.size(0)).long().cuda() # 각 샘플이 자기 자신과 가장 유사해야한다는 가정을 따름
        loss = F.cross_entropy(similarity_matrix, labels)
                        
        # 손실 및 시간 기록
        loss_mtr.update(loss.item(), image.shape[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 메모리 관리: 사용하지 않는 텐서 삭제 및 캐시 정리
        # del similarity_matrix, image_emb, audio_emb, labels

        batch_time.update(time.time() - end)
        end = time.time()

        global_step += 1

        # 미니 배치가 끝날 때마다 메모리 캐시 비우기
        torch.cuda.empty_cache()
        writer.add_scalar('Train/Loss(step)', loss.item(), global_step)


    # TensorBoard에 손실 및 시간 기록 추가
    writer.add_scalar('Train/Loss(epoch)', loss.item(), epoch)
    writer.add_scalar('Train/Average Loss', loss_mtr.avg, epoch)
    writer.add_scalar('Train/Batch Time', batch_time.avg, epoch)
    writer.add_scalar('Train/Data Loading Time', data_time.avg, epoch)


# def train(train_loader, model, optimizer, epoch, args, writer):
#     print("train 들어옴")
#     model.train() # 모델을 훈련 모드로 전환
#     batch_time = AverageMeter('Time', ':6.3f') # 소수점 이하 3자리까지, 총 6자리
#     data_time = AverageMeter('Data', ':6.3f') # 소수점 이하 3자리까지, 총 6자리
#     loss_mtr = AverageMeter('Loss', ':.3f')
#     recall_at_10_mtr = AverageMeter('Recall@10', ':.3f')


#     progress = ProgressMeter(
#         len(train_loader),
#         [batch_time, data_time, loss_mtr],
#         prefix="Epoch: [{}]".format(epoch),
#     )

#     end = time.time()
#     global_step = 0

#     if args.resume:
#         # 진행 상태 복원
#         exp_dir = args.exp_dir
#         progress_pkl = f"{exp_dir}/progress.pkl"
#         progress_data, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
#         print("\nResume training from:")
#         print("  epoch = %s" % epoch)
#         print("  global_step = %s" % global_step)
#         print("  best_epoch = %s" % best_epoch)
#         print("  best_acc = %.4f" % best_acc)
#     else:
#         epoch = 0  # 새로운 학습 시작 시 epoch을 0으로 설정

#     batch_size = 50

#     for i, (image, spec, _, _, _) in tqdm(enumerate(train_loader), desc="Train Embedding Extraction", total=len(train_loader)):
#         data_time.update(time.time() - end)
        
#         # import pdb; pdb.set_trace()

#         # 데이터 GPU로 이동
#         spec = spec.cuda()
#         image = image.cuda()

#         # import pdb; pdb.set_trace()

#         A,logits,Pos,Neg = model(image.float(), spec.float(), args, mode = 'train')
#         # logits.size() = [50, 52]
#         targets = torch.zeros(logits.size(0), dtype=torch.long).cuda()  # Positive class: index 0
#         # targets.size() = [50]
#         loss = F.cross_entropy(logits, targets)

#         # Get the indices of the top-k predictions for each sample
#         # import pdb; pdb.set_trace()
#         _, topk_indices = torch.topk(logits, k=10, dim=1)  # (B, top_k)
#         correct = (topk_indices == targets.unsqueeze(1)).sum(dim=1)  # (B,)
    
#         # Recall is the mean of correct predictions across the batch
#         recall_at_10 = correct.float().mean().item()
                
#         # 손실 및 시간 기록
#         loss_mtr.update(loss.item(), image.shape[0])
#         recall_at_10_mtr.update(recall_at_10, image.shape[0])
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # 메모리 관리: 사용하지 않는 텐서 삭제 및 캐시 정리
#         # del similarity_matrix, image_emb, audio_emb, labels
#         torch.cuda.empty_cache()

#         batch_time.update(time.time() - end)
#         end = time.time()

#         # TensorBoard에 손실 및 시간 기록 추가
#         writer.add_scalar('Train/Loss', loss.item(), global_step)
#         writer.add_scalar('Train/Average Loss', loss_mtr.avg, global_step)
#         writer.add_scalar('Train/Batch Time', batch_time.avg, global_step)
#         writer.add_scalar('Train/Data Loading Time', data_time.avg, global_step)

#         global_step += 1

#         # 미니 배치가 끝날 때마다 메모리 캐시 비우기
#         del loss
#         torch.cuda.empty_cache()


# 최종 느낌 validate 이었음
def validate(test_loader, model, args):
    model.cuda()
    model.eval()
    
    # 유사도 계산을 위한 평가 도구 초기화
    image_embeddings = []
    audio_embeddings = []
    ids = []
    # evaluator = utils.Evaluator()

    # 데이터셋에서 이미지와 오디오 임베딩 추출
    for step, (image, spec, _, name, _) in tqdm(enumerate(test_loader), desc="Validate Embedding Extraction", total=len(test_loader)):
        image, spec = image.cuda().float(), spec.cuda().float()

        with torch.no_grad():
            img_emb, aud_emb = model.extract_features(image, spec)
            
            # 이미지와 오디오 임베딩이 4차원일 경우 평균 풀링 적용 (B x 512 x W x H -> B x 512)
            if img_emb.dim() == 4:
                img_emb = F.avg_pool2d(img_emb, kernel_size=(img_emb.size(2), img_emb.size(3))).squeeze()
            if aud_emb.dim() == 4:
                aud_emb = F.avg_pool2d(aud_emb, kernel_size=(aud_emb.size(2), aud_emb.size(3))).squeeze()

        # 추출된 이미지와 오디오 임베딩 추가
        image_embeddings.append(img_emb)
        audio_embeddings.append(aud_emb)
        ids.extend(name)

    # 텐서로 결합
    image_embeddings = torch.cat(image_embeddings, dim=0)
    audio_embeddings = torch.cat(audio_embeddings, dim=0)

    # 유사도 행렬 계산 (블록 단위)
    similarity_matrix = torch.zeros(image_embeddings.size(0), audio_embeddings.size(0))
    batch_size = 128

    for i in range(0, image_embeddings.size(0), batch_size):
        for j in range(0, audio_embeddings.size(0), batch_size):
            img_batch = image_embeddings[i:i + batch_size].cuda()
            aud_batch = audio_embeddings[j:j + batch_size].cuda()
            similarity_matrix[i:i + batch_size, j:j + batch_size] = torch.mm(img_batch, aud_batch.T).cpu()
    # print(similarity_matrix.size()) # [1000, 1000]

    # Recall@10 계산
    # import pdb; pdb.set_trace()
    recall_at_10 = 0
    labels = torch.arange(similarity_matrix.size(0)).long() # 각 샘플이 자기 자신과 가장 유사해야한다는 가정을 따름
    # labels = [0, 1, 2,..., 999]
    _, topk_indices = similarity_matrix.topk(10, dim=1, largest=True, sorted=True) # 상위 10개의 예측 결과와 인덱스
    # topk_indices.size() = [1000, 10]
    recall_at_10 = (topk_indices == labels.unsqueeze(1)).sum().item() / labels.size(0)

    return recall_at_10


# def validate(test_loader, model, args):
#     """
#     Validate the model on the test dataset.

#     Args:
#         test_loader (DataLoader): DataLoader for the test set.
#         model (nn.Module): Trained model.
#         args (argparse.Namespace): Arguments with necessary configurations.

#     Returns:
#         recall_at_10 (float): Recall@10 metric.
#     """
#     model.cuda()
#     model.eval()

#     recall_at_10 = 0  # Recall@10 초기화
#     total_samples = 0  # 총 샘플 수

#     with torch.no_grad():  # 검증에서는 그래디언트 계산 비활성화
#         for step, (image, spec, _, _, _) in tqdm(enumerate(test_loader), desc="Validate Embedding Extraction", total=len(test_loader)):
#             # Move inputs to GPU

#             image, spec = image.cuda().float(), spec.cuda().float()

#             # Forward pass
#             A, logits, Pos, Neg = model(image, spec, args, mode='val')
#             # logits.size() = (1, 3)
#             # Pos.size() = (1, 1, 14, 14)

#             # Calculate Recall@10
#             batch_size = logits.size(0) # (1, 3)
#             # print(logits.size()) # [128, 130]
#             _, topk_indices = torch.topk(logits, k=10, dim=1)  # 상위 10개의 인덱스 가져오기
#             targets = torch.zeros(batch_size, dtype=torch.long).cuda()  # Positive class는 index 0
            
#             # Top-k 인덱스에서 Positive Pair 확인
#             correct = (topk_indices == targets.unsqueeze(1)).sum(dim=1)  # (B,)
#             recall_at_10 += correct.float().sum().item()  # Recall@10 누적
#             total_samples += batch_size

#     # Normalize Recall@10
#     recall_at_10 /= total_samples

#     return recall_at_10



def evaluate_retrieval(similarity_matrix, ids):
    top_k = 5  # Top-K 정확도 평가 예시
    correct_top_k = 0

    for i in range(len(ids)):
        sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
        top_k_indices = sorted_indices[:top_k]

        if i in top_k_indices:
            correct_top_k += 1

    accuracy_top_k = correct_top_k / len(ids)
    print(f'Top-{top_k} Retrieval Accuracy: {accuracy_top_k:.4f}')
    return accuracy_top_k



class AverageMeter(object):
    # 평균값을 계산하고 저장하는 도우미 class
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    # 훈련과정에서 진행상황을 출력하기 위한 클래스
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters # 추적하고자하는
        self.prefix = prefix
        self.fp = fp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main(get_arguments())
