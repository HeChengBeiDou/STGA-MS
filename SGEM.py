import torch
import torch.nn as nn

class SGEM(nn.Module):
    def __init__(self, n_channels, h, w):
        super(SGEM, self).__init__()
        self.n_channels= n_channels

        self.conv2d = nn.Conv2d(h*w, h*w, kernel_size=3, stride=1, bias=False, padding=1, groups=w)
        
        self.conv2d_2 = nn.Conv2d(self.n_channels, self.n_channels,kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        
        # self.pool = nn.AdaptiveMaxPool2d(1)
        # self.fc = nn.Linear(self.n_channelsself.n_channels
    def forward(self, x):
        '''
        x: shape (n, t, c, h, w)
        '''
        assert len(x.size()) == 5
        n, t, c, h, w = x.size()
        
        # x = x.view(n*t, c, h, w)
        x_permute = x.permute(0, 4, 3, 1, 2).contiguous().view(n, -1, t, c)#(-1, 1, t, c)
        
        x_permute = self.conv2d(x_permute)#(n, -1, t, c)
        ##  如果把二维卷积用在c t维度上  
        ##  也可以尝试让分组卷积 每个channel的t帧一组 按行分组
        
        x_permute = x_permute.view(n, h, w, t, c).permute(0, 3, 4, 1, 2)#(n, t, c, h, w)
        
        # x = x.view(n*t, c, h, w)
        # x_conv = self.conv2d_2(x)
        # x = x.view(n, t, c, h, w)
        # x_conv = x_conv.view(n, t, c, h, w)
        # x = x + x_permute

        return x_permute

if __name__ == '__main__':
    a = SGEM(16, 224, 224)
    data = torch.zeros(3, 3, 16, 224, 224)
    out = a(data)
    print(out.shape)
        
        
        
        
        
        
        
        
        
            
            