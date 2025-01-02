import torch
from torch import nn
import torch.nn.functional as F
from models import audio_convnet
from models import image_convnet
from models import base_models
import pdb
import random

def normalize_img(value, vmax=None, vmin=None):
    #  pdb.set_trace()
    value1 = value.view(value.size(0), -1)
    value1 -= value1.min(1, keepdim=True)[0]
    value1 /= value1.max(1, keepdim=True)[0]
    return value1.view(value.size(0), value.size(1), value.size(2), value.size(3))

class AVENet(nn.Module):

    def __init__(self, args):
        super(AVENet, self).__init__()

        # -----------------------------------------------
        self.imgnet = base_models.resnet18(modal='vision', pretrained=True)
        self.audnet = base_models.resnet18(modal='audio')
        self.imgnet = self.imgnet.cuda()
        self.audnet = self.audnet.cuda()
        self.img_proj = nn.Linear(512, 512)
        self.aud_proj = nn.Linear(512, 512)

        self.m = nn.Sigmoid()
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        self.epsilon = args.epsilon
        self.epsilon2 = args.epsilon2
        self.tau = 0.03
        self.trimap = args.tri_map
        self.Neg = args.Neg
        self.random_threshold = args.random_threshold

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, image, audio,args,mode='val'):
        # Image
        image = image.float().cuda()
        audio = audio.float().cuda()
        # import pdb; pdb.set_trace()

        B = image.shape[0] # batch_size
        self.mask = ( 1 -100 * torch.eye(B,B)).cuda()
        img_feature = self.imgnet(image)
        img =  nn.functional.normalize(img_feature, dim=1) # (B, 512, 14, 14)

        # Audio
        aud_feature = self.audnet(audio)
        aud = self.avgpool(aud_feature).view(B,-1)
        aud = nn.functional.normalize(aud, dim=1) # (B, 512)
        # Join them
        # A = torch.einsum('ncqa,nchw->nqa', [img, aud.unsqueeze(2).unsqueeze(3)]).unsqueeze(1) # (B, 1, 14, 14)
        A0 = torch.einsum('ncqa,ckhw->nkqa', [img, aud.T.unsqueeze(2).unsqueeze(3)]) # (B, B, 14, 14)

        # trimap
        # Pos = self.m((A - self.epsilon)/self.tau) # Pos.size() = (B, 1, 14, 14)
        # if self.trimap:  # True
        #     Pos2 = self.m((A - self.epsilon2)/self.tau) 
        #     Neg = 1 - Pos2
        # else:
        #     Neg = 1 - Pos

        Pos_all =  self.m((A0 - self.epsilon)/self.tau) # (B, B, 14, 14)

        # positive
        # sim1 = (Pos * A).view(*A.shape[:2],-1).sum(-1) / (Pos.view(*Pos.shape[:2],-1).sum(-1)) # (B, 1)
        #negative
        # sim = ((Pos_all * A0).view(*A0.shape[:2],-1).sum(-1) / Pos_all.view(*Pos_all.shape[:2],-1).sum(-1) )* self.mask # (B, B)
        sim = ((Pos_all * A0).view(*A0.shape[:2],-1).sum(-1) / Pos_all.view(*Pos_all.shape[:2],-1).sum(-1) ) # (B, B)
        
        # sim2 = (Neg * A).view(*A.shape[:2],-1).sum(-1) / Neg.view(*Neg.shape[:2],-1).sum(-1) # (B, 1)

        # if self.Neg: # True
        #     logits = torch.cat((sim1,sim,sim2),1)/0.07 # (B, B+2)
        # else:
        #     logits = torch.cat((sim1,sim),1)/0.07

        image_emb = self.avgpool(img_feature).view(img_feature.size(0), -1)
        image_emb = self.img_proj(image_emb)
        image_emb = nn.functional.normalize(image_emb, dim=1)

        audio_emb = self.avgpool(aud_feature).view(aud_feature.size(0), -1)
        audio_emb = self.aud_proj(audio_emb)
        audio_emb = nn.functional.normalize(audio_emb, dim=1)


        return image_emb, audio_emb, sim
        #return A,logits,Pos,Neg


    # feature extraction function 구현
    def extract_features(self, image, audio):
        """
        Returns intermediate image and audio features without calculating logits or other outputs.
        """
        # with torch.no_grad():
            # Extract features from image and audio networks
        image = image.float().cuda() # 여기 뒤에 cuda() 붙이는거 우선 보류
        audio = audio.float().cuda()

        image_feature = self.imgnet(image)  # Intermediate image features
        audio_feature = self.audnet(audio)  # Intermediate audio features
        
        # Normalize the features if needed
        image_feature = self.avgpool(image_feature).view(image_feature.size(0), -1)
        image_feature = nn.functional.normalize(image_feature, dim=1)
        audio_feature = self.avgpool(audio_feature).view(audio_feature.size(0), -1)
        audio_feature = nn.functional.normalize(audio_feature, dim=1)

        return image_feature, audio_feature