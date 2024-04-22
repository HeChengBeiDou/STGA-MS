# 主模型
import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import cv2

try:
    from model.net.attention_A import attention_A
except:
    from attention_A import attention_A

class resnet_modify(nn.Module):
    def __init__(self,n_classes, in_frames, n_channels, model_path=None, h=224, w=224):
        super(resnet_modify, self).__init__()
        self.in_frames = in_frames
        self.n_channels= n_channels
        self.n_classes = n_classes

        self.resnet_stga_multiscale = torchvision.models.resnet18(pretrained=True)
        # self.resnet_stga_multiscale.requires_grad_(False)#只更新 新增加的参数
        
        self.resnet_stga_multiscale.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        
        self.resnet_stga_multiscale.layer1 = self.modify_block(layer=self.resnet_stga_multiscale.layer1,n_channels = 64,h=h//4, w=w//4)
        self.resnet_stga_multiscale.layer2 = self.modify_block(layer=self.resnet_stga_multiscale.layer2,n_channels = 64,h=h//4, w=w//4)
        self.resnet_stga_multiscale.layer3 = self.modify_block(layer=self.resnet_stga_multiscale.layer3,n_channels = 128,h=h//8, w=w//8)
        self.resnet_stga_multiscale.layer4 = self.modify_block(layer=self.resnet_stga_multiscale.layer4,n_channels = 256,h=h//16, w=w//16)

        
        self.block_1_stream = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(False),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                     nn.ReLU(False),
                                    nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(False),
                                     )
        self.block_2_stream = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(False),
                                    nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                     nn.ReLU(False),
                                     )        
        self.block_3_stream = nn.Sequential(
                                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(False),
                                     )
        self.avgpool = nn.AdaptiveAvgPool2d(1)        
        self.fc = nn.Linear(768, 10)
        
    def multi_scale_block(self, x):
        n, t, c, h, w = x.size()
        x_permute = x.view(n * t, c, h, w)
        
        x_1_1 = self.block_1_stream[0:3](x_permute)
        x_2_1 = self.block_2_stream[0:3](x_1_1)
        x_3_1 = self.block_3_stream[0:3](x_2_1)#出
        
        x_1_2 = self.block_1_stream[3:6](x_1_1)
        x_2_1 = self.block_2_stream[0:3](x_1_1)
        
        x_1_3 = self.block_1_stream[6:9](torch.cat((x_1_2, x_2_1), 1)) #出
        x_2_2 = self.block_2_stream[3:6](torch.cat((x_2_1, x_1_2), 1)) #出

        out = torch.cat((x_1_3, x_2_2, x_3_1), 1)#(n*t, c, h, w)
        return out
        
    def modify_block(self, layer, n_channels, h, w):
        tmp_layers = []
        for idx, item in enumerate(layer):
            if idx == 0:
                # h_list = [56, 28, 14, 7]
                # w_list = [56, 28, 14, 7]
                # channel_list = [64, 128, 256, 512]
                tmp_layers.append(attention_A(n_channels=n_channels, h=h, w=w, in_frames=self.in_frames))
                tmp_layers.append(item)
            else:
                tmp_layers.append(item)
        out_layer = nn.Sequential(*tmp_layers)
        return out_layer        

 
    def forward(self, x):
        '''
        x: shape (n, t, c, h, w)
        '''
        assert len(x.size()) == 5
        n, t, c, h, w = x.size()
        x = x.view(n*t, c, h, w)
        x = self.resnet_stga_multiscale.conv1(x)
        x = self.resnet_stga_multiscale.bn1(x)
        x = self.resnet_stga_multiscale.relu(x)
        x = self.resnet_stga_multiscale.maxpool(x)


        for layer in self.resnet_stga_multiscale.layer1:
            if not isinstance(layer, attention_A):
                x = layer(x)#在这里面加模块 这里需要for循环 如果是attention需要改成五维的
            else:
                x = x.view(n, t, -1, x.size(2), x.size(3))
                x = layer(x)
                x = x.view(n*t, -1, x.size(3), x.size(4))
        
        ## 这里加多尺度 和下面的stack出来的结果堆叠
        x = x.view(n, t, -1, x.size(2), x.size(3))
        x_scale = self.multi_scale_block(x)
        x = x.view(n*t, -1, x.size(3), x.size(4))#
        
        for layer in self.resnet_stga_multiscale.layer2:
            if not isinstance(layer, attention_A):
                x = layer(x)#第一次64-128 然后下面的
            else:
                x = x.view(n, t, -1, x.size(2), x.size(3))
                x = layer(x)
                x = x.view(n*t, -1, x.size(3), x.size(4))
        for layer in self.resnet_stga_multiscale.layer3:
            if not isinstance(layer, attention_A):
                x = layer(x)#在这里面加模块 这里需要for循环 如果是attention需要改成五维的
            else:
                x = x.view(n, t, -1, x.size(2), x.size(3))
                x = layer(x)
                x = x.view(n*t, -1, x.size(3), x.size(4))
        for layer in self.resnet_stga_multiscale.layer4:
            if not isinstance(layer, attention_A):
                x = layer(x)#在这里面加模块 这里需要for循环 如果是attention需要改成五维的
            else:
                x = x.view(n, t, -1, x.size(2), x.size(3))
                x = layer(x)
                x = x.view(n*t, -1, x.size(3), x.size(4))
        
        # x = x.view(n* t, -1)
        x = torch.cat((x_scale, x),1)


        x = self.avgpool(x)
        x = x.view(n*t, -1)
        x = self.fc(x)
        x = x.view(n, t, -1)

        return x        



if __name__ == '__main__':
    device_ids = [3]
    device = torch.device("cuda:{}".format(
    device_ids[0]) if torch.cuda.is_available() else "cpu")

    in_frames = 15
    n_channels = 2
    n_classes = 2
    batch_size = 2
    model = resnet_modify(n_classes=n_classes, in_frames=in_frames,n_channels=n_channels).to(device)
    from utils.model_evaluate import getModelSize
    getModelSize(model)
    for name, parameters in model.named_parameters():  
        print(name, ';', parameters.requires_grad, parameters.size())
    
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    # 这一句不能少
    model.to(device=device)
    x = torch.zeros(batch_size, in_frames, 2, 224, 224).to(device)
    
    out = model(x)
    print(out.shape)  #
