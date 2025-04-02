import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from DatasetLoader_origin import GetAudioVideoDataset
from tqdm import tqdm
import random

import json
import time

# 데이터셋 경로와 파라미터 설정
FEATURES_FILE = './top_k/topk_features/features_vggsx512_60000.pt'
SIMILARITY_FILE = './top_k/topk_similarity/similarity_matrix_60000x60000_60000.pt'
TOP_K_FILE = './top_k_similarity_60000.json'
BATCH_SIZE = 16
TOP_K_VALUES = [60000]

import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Dataset paths and configurations
    parser.add_argument('--trainset', type=str, default='vggss', help='Name of the training dataset (e.g., flickr, vggss)')
    parser.add_argument('--testset', type=str, default='vggss', help='Name of the test dataset (e.g., flickr, vggss)')
    parser.add_argument('--image_size',default=224,type=int,help='Height and width of inputs')

    parser.add_argument('--multiprocessing_distributed', action='store_true', help='Use multi-processing distributed training')

    return parser.parse_args()

# args 초기화
args = get_args()


# ----------1. ResNet18 모델 준비---------- #
print("Loading ResNet18 model...")
model = resnet18(pretrained=True)
# model = nn.Sequential(*list(model.children())[:-2], nn.AdaptiveAvgPool2d(1))
# # Resnet18에서 fc layer랑 마지막 avgpool layer를 제외한 부분만 가져옴
# # nn.AdaptiveAvgPool2d(1)은 마지막 avgpool layer를 대체함

model = nn.Sequential(*list(model.children())[:-1])  # fc layer만 제거, avgpool 유지

model = model.eval().cuda()

# ----------2. 데이터 로더 준비---------- #
print("Preparing DataLoader...")

# Train dataset 생성
traindataset = GetAudioVideoDataset(args = args, mode='train')  # Train dataset 생성


# Train loader 설정
train_loader = torch.utils.data.DataLoader(
    traindataset,
    batch_size=16,
    shuffle= False, 
    num_workers=1,  # 데이터 로딩 작업 스레드 수
)

print("DataLoader ready.")

# -----------3. Feature 추출---------- #
print("Extracting features...")
features = torch.zeros((len(traindataset), 512), device='cuda')
# len(traindataset) = 61142

start_time = time.time()
with torch.no_grad():
    id_list = []
    for idx, (frames, _, _, video_id, _) in tqdm(enumerate(train_loader), desc="Extracting features", total=len(train_loader)):
        # import pdb; pdb.set_trace()

        frames = frames.cuda()  # 이미지만 사용, frames.size() = (16, 3, 224, 224)

        # 각 프레임에 전처리 적용
        for id in video_id:
            id_list.append(id)
        
        batch_features = model(frames).view(frames.size(0), -1)  # (B, 512) = (16, 512)
        batch_features = F.normalize(batch_features, p=2, dim=1)  # L2 Unit Norm, size = (16, 512)
        start_idx = idx * BATCH_SIZE
        end_idx = start_idx + frames.size(0)
        features[start_idx:end_idx] = batch_features

print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds.")
# torch.save(features.cpu(), FEATURES_FILE)

# import pdb; pdb.set_trace()
print("id_list len : ", len(id_list))  # 61142여야 하는데 61136이 나옴
print(id_list)

# ----------4. Similarity Matrix 계산---------- #
print("Calculating similarity matrix row by row...")

# CPU로 feature 이동
features = features.cpu()

# Similarity Matrix를 파일에 직접 저장할 준비
num_samples = features.size(0) # 61142
similarity_matrix = torch.zeros((num_samples, num_samples))

# 각 이미지에 대해 유사도 계산
start_time = time.time()
for i in range(num_samples):
    # i번째 이미지와 나머지 모든 이미지 간의 유사도 계산
    similarity_row = torch.mm(features[i:i+1], features.T)  # (1, N)
    similarity_matrix[i] = similarity_row  # 결과 저장

    if i % 100 == 0:  # 진행 상태 출력
        print(f"Processed {i}/{num_samples} images...")

# Similarity Matrix 저장
# torch.save(similarity_matrix, SIMILARITY_FILE)
print(f"Similarity matrix saved. Total time: {time.time() - start_time:.2f} seconds.")


# 5. Top-K 유사도 계산 및 저장
print("Calculating Top-K similarity...")
top_k_results = {}

num_samples = similarity_matrix.size(0)


for i in range(num_samples):
    similarities = similarity_matrix[i]
    video_id = id_list[i]

    for k in TOP_K_VALUES:
        # Top-K 추출
        topk_values, topk_indices = torch.topk(similarities, k + 1)  # 자기 자신 포함
        # topk_values = topk_values[1:].tolist()  # 자기 자신 제외
        topk_indices = topk_indices[1:].tolist()  # 자기 자신 제외

        if video_id not in top_k_results:
            top_k_results[i] = {}

        random_index = random.choice(topk_indices)  # 텐서를 리스트로 변환 후 random.choice 사용
        random_index = [random_index]

        top_k_results[i][f"Top-{k}"] = {"video_id" : video_id, "indices": random_index}
        # top_k_results[i][f"Top-{k}"] = {"video_id" : video_id, "indices": topk_indices}
        # top_k_results[video_id][f"Top-{k}"] = {"indices": topk_indices}


# JSON 파일로 저장
with open(TOP_K_FILE, 'w') as f:
    json.dump(top_k_results, f)

print("Top-K similarity saved.")
