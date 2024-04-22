# 主模型
import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import torchvision


# try:
#     from model.net.resnet import resnet18
# except:
#     from resnet import resnet18
    
try:
    from model.net.resnet_modify import resnet_modify
except:
    from resnet_modify import resnet_modify

class FinalModule(nn.Module):
    def __init__(self, in_frames=15, n_channels=2, n_classes=6, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, area=32):
        super(FinalModule, self).__init__()
        self.in_frames = in_frames
        self.n_channels= n_channels
        self.n_classes = n_classes

        self.resnet_stga = resnet_modify(n_classes=n_classes, in_frames=in_frames,n_channels=n_channels)# 先加载权重，然后再改 (n, t, -1)
        
        # for param in self.resnet_stga.parameters():
        #     param.requires_grad=False

        self.patch_conv_stream = nn.Sequential(
            nn.Conv2d(self.n_classes, 64, kernel_size=3, stride=2, padding=0, bias=False),nn.BatchNorm2d(64),
                                     nn.ReLU(False),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0, bias=False),
                                    nn.AdaptiveAvgPool2d(1)
                                     )
        self.fc_out = nn.Linear(29,n_classes)
        
        self.fc_stga_m = nn.Linear(self.in_frames,1)#去掉时间维度
        
        self.lstm_p = nn.LSTM(
            input_size=self.in_frames,
            hidden_size=1,
            num_layers=3,
            batch_first=True,# input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        
        self.lstm_p_patch = nn.LSTM(
            input_size=self.in_frames,
            hidden_size=1,        
            num_layers=3,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        
        self.fc_p_patch = nn.Linear(73, 10)
        

    def single_segment(self, x):#这里改成多尺度
        '''
        x (n, t, c, h, w)
        '''
        n, t, c, h, w = x.size()
        # x = x.view(n * t, c, h, w)
        x_out = self.resnet_stga(x)#(n, t, -1)
        return x_out
    
    def patch_single_segment(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n*t, c, h, w)
        x = self.patch_conv_stream(x)
        x = x.view(n, t, -1)
        return x
        
    def forward(self, x, p, patch):
        '''
        x1 : (n, t, c, h, w)
        p1 : (n, t, 24)
        '''
        n, t, c, h, w = x.size()
        
        # x1 = x1.view(n * t, c, h, w)
        
        x_stga = self.resnet_stga(x)
        x_stga = x_stga.permute(0, 2, 1)
        x_stga = self.fc_stga_m(x_stga).view(n,-1)
        
        ## 加邻域信息
        patch = self.patch_single_segment(patch)#(n, t, 64)
        
        p_lstm = p.permute(0, 2, 1)
        
        # self.lstm_p.flatten_parameters()
        p_lstm, (h_n, h_c) = self.lstm_p(p_lstm, None) #(n, c, t)->(n,c,1)  中间那个
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. p has shape=(batch, time_step, output_size) """
        p_lstm = p_lstm.view(n, -1)
        
        p_patch = torch.cat((p, patch), dim=2)#(n, t, -1)
        
        p_patch = p_patch.permute(0, 2, 1)
        
        p_patch_lstm, (h_n, h_c) = self.lstm_p_patch(p_patch, None)#(n, 1, -1)
        p_patch_lstm = p_patch_lstm.view(n, -1)
        
        p_patch_lstm = self.fc_p_patch(p_patch_lstm)
        
        x_out = torch.cat((x_stga, p_lstm, p_patch_lstm), dim=1)
        
        x_out = self.fc_out(x_out)
        return x_out


if __name__ == '__main__':
    device_ids = [3, 4]
    device = torch.device("cuda:{}".format(
    device_ids[0]) if torch.cuda.is_available() else "cpu")

    in_frames = 15
    batch_size = 2
    model = FinalModule(n_classes=1, in_frames=in_frames).to(device)
    # import sys
    # sys.path.append(r"../../")
    # from utils.model_evaluate import *
    # getModelSize(model)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    # 这一句不能少
    model.to(device=device)
    x1 = torch.zeros(batch_size, in_frames, 2, 224, 224).to(device)

    p1 = torch.zeros(batch_size, in_frames, 9).to(device)

    patch1 = torch.zeros(batch_size, in_frames, 1, 64, 64).to(device)

    
    out = model(x1, p1, patch1)
    print(out.shape)  #
