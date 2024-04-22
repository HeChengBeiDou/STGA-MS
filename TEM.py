import torch
import torch.nn as nn

class TEM(nn.Module):
    def __init__(self, n_channels, in_frames):
        super(TEM, self).__init__()
        self.n_channels= n_channels
        self.in_frames = in_frames
        
        self.conv2d_1 = nn.Conv2d(self.n_channels, self.n_channels,kernel_size=1, stride=1, bias=False, padding=0, groups=1)
        self.conv2d_2 = nn.Conv2d(self.n_channels, self.n_channels,kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        self.conv2d_3 = nn.Conv2d(self.n_channels, self.n_channels,kernel_size=1, stride=1, bias=False, padding=0, groups=1)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(self.in_frames, self.in_frames)
        
    def forward(self, x):
        '''
        x: shape (n, t, c, h, w)
        '''
        assert len(x.size()) == 5

        n, t, c, h, w = x.size()
        x_reshape_1 = x.view(n*t, c, h, w)
        x_conv2d_1 = self.conv2d_1(x_reshape_1) # (n*t, c, h, w)
        x_conv2d_1_minus0, _ = x_conv2d_1.view(n, t, c, h, w).split([t-1, 1], dim=1)
        _, x_conv2d_1_minus1 = x_conv2d_1.view(n, t, c, h, w).split([1, t-1], dim=1)
        
        x_conv2d_1_minus0 = x_conv2d_1_minus0.contiguous().view(-1, c, h, w)
        x_conv2d_1_minus1 = x_conv2d_1_minus1.contiguous().view(-1, c, h, w)
        
        x_conv2d_1_minus0_conv2d_2 = self.conv2d_2(x_conv2d_1_minus0)
        x_conv2d_1_minus1_conv2d_2 = self.conv2d_2(x_conv2d_1_minus1)
        
        x_conv2d_1_minus0_conv2d_2 = x_conv2d_1_minus1_conv2d_2.contiguous().view(n, t-1, c, h, w)
        x_conv2d_1_minus1_conv2d_2 = x_conv2d_1_minus1_conv2d_2.contiguous().view(n, t-1, c, h, w)
        
        x_sep = x_conv2d_1_minus1_conv2d_2 - x_conv2d_1_minus0_conv2d_2
        
        tmp = torch.zeros(n, 1, c, h, w).to(device=x_sep.device)
        x_integrate = torch.cat((x_sep, tmp), dim=1)  # (n, t, c, h, w)
        x_integrate = x_integrate.contiguous().view(n*t, c, h, w)
        out = self.conv2d_3(x_integrate) #(n*t,c,h,w)
        
        out_1 = out.view(n, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous().view(n*c, t, h, w)
        out_1 = self.pool(out_1).view(n, c, t)
        out_1 = self.fc(out_1).view(n, c, t, 1, 1).permute(0, 2, 1, 3, 4)#(n, t, c, 1, 1)
        out_1 = out_1.contiguous().view(n*t, c, 1, 1)
        out = out_1 * out
        
        out = out.view(n, t, c, h, w)
        # out = out + x
        return out

if __name__ == '__main__':
    a = TEM(16)
    data = torch.zeros(3, 3, 16, 224, 224)
    out = a(data)
    print(out.shape)