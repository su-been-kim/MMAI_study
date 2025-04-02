import os
import cv2
import json
import torch
import csv
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb
import time
from PIL import Image
import glob
import sys 
import scipy.io.wavfile as wav
from scipy import signal
import random
import soundfile as sf

class GetAudioVideoDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None):
        if mode == 'train':
            json_file = '/mnt/scratch/users/individuals/VGGsound_individual/metadata/train_a_third.json'
            data_path = '/mnt/scratch/users/individuals/VGGsound_individual/train'
            # json_file = '/mnt/scratch/users/sally/VGGsound_individual/metadata/train_practice.json'
            # data_path = '/mnt/scratch/users/sally/VGGsound_individual/train_practice'
        else:
            json_file = '/mnt/scratch/users/individuals/VGGsound_individual/metadata/test.json'
            data_path = '/mnt/scratch/users/individuals/VGGsound_individual/test'

        with open(json_file, 'r') as f:
            self.data = json.load(f)['data']
            
            if mode == 'test':
                self.data = self.data[:1000]

        self.audio_path = os.path.join(data_path, 'sample_audio')
        self.video_path = os.path.join(data_path, 'sample_frames/frame_4')

        
        self.imgSize = args.image_size 
        self.mode = mode
        self.transforms = transforms
        self.topk = args.topk

        # initialize video transform
        self._init_atransform()  # audio spectrogram의 텐서화 및 정규화를 수행
        self._init_transform() # image 변환 작업을 수행
        #  Retrieve list of audio and video files

        # JSON 데이터 로드 (for top-k similarity)
        with open(f'/mnt/scratch/users/sally/top_k_similarity_{args.topk}.json', 'r') as f:
            self.top_k_data = top_k_data = json.load(f)

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.mode == 'train': # 훈련 모드
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC), # image를 self.imgSize의 1.1배 크기로 확대
                transforms.RandomCrop(self.imgSize), # random crop을 통해 image의 일부분을 임의로 자름
                transforms.RandomHorizontalFlip(), # 50% 확률로 image를 좌우로 뒤집어 데이터 다양성을 더함.
                transforms.CenterCrop(self.imgSize), # 중앙 부분을 잘라냄
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]) # 텐서로 변환 후 정규화
        else: # test 모드
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]) # 텐서로 변환 후 정규화           

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])
        # audio spectrogram data를 tensor 형식으로 변환. -> 표준 편차를 12.0으로 설정하여 정규화
  

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB') # 주어진 경로에서 image를 읽고 RGB로 변환하여 반환
        return img

    def __len__(self):
        # Consider all positive and negative examples
        return len(self.data)  # self.length

    def __getitem__(self, idx):
        # json file에서 audio & video path 가져오기
    
        item = self.data[idx]
        video_id = item['video_id']

        audio_path = os.path.join(self.audio_path, f"{video_id}.wav")
        video_path = os.path.join(self.video_path, f"{video_id}.jpg")

        # ======= 1. 기본적인 image, audio 데이터 가져오기 ======= # 

        # image load 및 전처리
        frame = self.img_transform(self._load_frame(video_path))
        frame_ori = np.array(self._load_frame(video_path))
        
        # 오디오 로드 및 전처리
        samples, samplerate = sf.read(audio_path)
        
        audio_length = samples.shape[0]
        three_sec_length = samplerate * 3  # 3초 길이
        fixed_length = int(6.5 * samplerate)  # 6.5초 길이

        # repeat if audio is too short -> audio 길이가 6.5초에 미치지 못하면, 필요한 길이만큼 반복하여 samples를 확장
        if audio_length < fixed_length:
            n = int(fixed_length / samples.shape[0]) + 1
            samples = np.tile(samples, n)
        resamples = samples[:fixed_length] # samples에서 정확히 6.5초를 잘라내어 resamples에 저장

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        start_fixed = int(3.5 * samplerate) # 3.5초부터 시작
        end_fixed = int(6.5 * samplerate) # 6.5초까지 
        fixed_samples = resamples[start_fixed:end_fixed] # 3.5초부터 6.5초까지의 samples를 fixed_samples에 저장

        frequencies, times, spectrogram = signal.spectrogram(fixed_samples,samplerate, nperseg=512,noverlap=274) # audio spectrogram을 생성
        spectrogram = np.log(spectrogram+ 1e-7) # spectrogram의 값을 log scale로 변환 -> spectrogram에 저장
        spectrogram = self.aid_transform(spectrogram) # spectrogram data를 tensor로 변환하고 정규화


        # image semantic & augmentation
        random_spectrogram = torch.zeros(1, 1, dtype=torch.float32)
        v_hp_frame = torch.zeros((3, self.imgSize, self.imgSize))  # 기본값 설정
        v_aug_frame = torch.zeros((3, self.imgSize, self.imgSize))  # 기본값 설정

        if self.mode == 'train':
            # ======= 2. top-k similarity 데이터 가져오기 (Train 한정) ======= #
            indices = None
            topk_key = f"Top-{self.topk}"  # 동적으로 key 생성
            for key, value in self.top_k_data.items():
                if value[topk_key]["video_id"] == video_id:
                    indices = value[topk_key]["indices"]

            selected_index = random.choice(indices)
            v_hp_id = self.data[selected_index]["video_id"]
            v_hp_path = '/mnt/scratch/users/individuals/VGGsound_individual/train/sample_frames/frame_4'
            v_hp_path = os.path.join(v_hp_path, f"{v_hp_id}.jpg")
            v_hp_frame = self.img_transform(self._load_frame(v_hp_path))

            # ======= 3. data augmentation (img) ====== #
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            aug_img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                transforms.CenterCrop(self.imgSize),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

            v_aug_frame = aug_img_transform(self._load_frame(video_path))

            # ====== 4. data augmentation (audio) ====== #
            max_start_idx = resamples.shape[0] - three_sec_length  # 3초 샘플 시작 가능한 최대 인덱스
            start_ix = random.randint(0, max_start_idx)  # 시작 인덱스를 랜덤하게 선택
            random_samples = samples[start_ix:start_ix + three_sec_length]  # 3초 샘플을 random_samples에 저장


            random_samples[random_samples > 1.] = 1.
            random_samples[random_samples < -1.] = -1.

            frequencies, times, random_spectrogram = signal.spectrogram(
                random_samples, samplerate, nperseg=512, noverlap=274
            )
            random_spectrogram = np.log(random_spectrogram + 1e-7)
            random_spectrogram = self.aid_transform(random_spectrogram)


        return frame,spectrogram,random_spectrogram, v_hp_frame, v_aug_frame,resamples,video_id,torch.tensor(frame_ori)