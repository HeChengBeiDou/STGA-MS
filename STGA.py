import torch
import torch.nn as nn
try:
    from TEM import TEM
    from SGEM import SGEM
except:
    from model.net.TEM import TEM
    from model.net.SGEM import SGEM

class STGA(nn.Module):
    def __init__(self, n_channels, h, w, in_frames):
        super(STGA, self).__init__()
        self.n_channels= n_channels
        self.in_frames = in_frames

        self.TEM = TEM(self.n_channels, self.in_frames)
        self.SGEM = SGEM(self.n_channels, h, w)
        
        self.conv2d_1 = nn.Conv2d(self.n_channels, self.n_channels,kernel_size=1, stride=1, bias=False, padding=0, groups=1)
        self.conv2d_2 = nn.Conv2d(self.n_channels, self.n_channels,kernel_size=1, stride=1, bias=False, padding=0, groups=1)
        
    def forward(self, x):
        '''
        x: shape (n, t, c, h, w)
        '''
        assert len(x.size()) == 5
        n, t, c, h, w = x.size()
        x_reshape_1 = x.view(n*t, c, h, w)
        x_conv2d_1 = self.conv2d_1(x_reshape_1) # (n*t, c, h, w)
        
        x_conv2d_1 = x_conv2d_1.view(n, t, c, h, w)
        x_TEM = self.TEM(x_conv2d_1)
        x_SGEM = self.SGEM(x_conv2d_1)
        
        x_out = x_TEM + x_SGEM
        # x_out = torch.cat((x_TEM, x_SGEM), 2)
        
        x_out = x_out.view(n*t, c, h, w)

        x_out = self.conv2d_2(x_out)
        x_out = x_out.view(n, t, c, h, w)
        out = x_out + x
        return out

if __name__ == '__main__':
    a = STGA(16)
    data = torch.zeros(3, 3, 16, 224, 224)
    out = a(data)
    print(out.shape)
        
        
        
        
        
        
        
        
        
            
            