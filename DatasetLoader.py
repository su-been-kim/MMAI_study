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

    def __init__(self, args, transforms=None):
        # Train 데이터 로드
        train_json_file = '/mnt/scratch/users/individuals/VGGsound_individual/metadata/train_a_third.json'
        train_data_path = '/mnt/scratch/users/individuals/VGGsound_individual/train'
        with open(train_json_file, 'r') as f:
            train_data = json.load(f)['data']

        # Test 데이터 로드
        test_json_file = '/mnt/scratch/users/individuals/VGGsound_individual/metadata/test.json'
        test_data_path = '/mnt/scratch/users/individuals/VGGsound_individual/test'
        with open(test_json_file, 'r') as f:
            test_data = json.load(f)['data']  # 테스트 데이터 5000개로 제한

        # 데이터 합치기
        self.data = train_data + test_data
        self.audio_paths = {
            "train": os.path.join(train_data_path, 'sample_audio'),
            "test": os.path.join(test_data_path, 'sample_audio')
        }
        self.video_paths = {
            "train": os.path.join(train_data_path, 'sample_frames/frame_4'),
            "test": os.path.join(test_data_path, 'sample_frames/frame_4')
        }

        # 데이터셋 구분 정보 추가
        self.data_modes = ['train'] * len(train_data) + ['test'] * len(test_data)

        # 기타 설정
        self.imgSize = args.image_size
        self.transforms = transforms
        self._init_atransform()
        self._init_transform()

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.img_transforms = {
            "train": transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            "test": transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0], std=[12.0])
        ])

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 데이터셋 모드에 따라 파일 경로 및 변환 로드
        mode = self.data_modes[idx]
        item = self.data[idx]
        video_id = item['video_id']

        audio_path = os.path.join(self.audio_paths[mode], f"{video_id}.wav")
        video_path = os.path.join(self.video_paths[mode], f"{video_id}.jpg")

        # Image 처리
        frame = self.img_transforms[mode](self._load_frame(video_path))
        frame_ori = np.array(self._load_frame(video_path))

        # Audio 처리
        samples, samplerate = sf.read(audio_path)
        if samples.shape[0] < samplerate * 10:
            n = int(samplerate * 10 / samples.shape[0]) + 1
            samples = np.tile(samples, n)
        resamples = samples[:samplerate * 10]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(
            resamples, samplerate, nperseg=512, noverlap=274
        )
        spectrogram = np.log(spectrogram + 1e-7)
        spectrogram = self.aid_transform(spectrogram)

        return frame, spectrogram, resamples, video_id, torch.tensor(frame_ori)