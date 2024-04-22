import torch
import torch.nn as nn

try:
    from model.net.STGA import STGA
except:
    from STGA import STGA
    
class attention_A(nn.Module):
    def __init__(self, n_channels, in_frames, h, w):
        super(attention_A, self).__init__()
        self.n_channels= n_channels
        self.in_frames = in_frames
        self.module = STGA(n_channels=self.n_channels,h=h, w=w, in_frames=self.in_frames)

        
    def forward(self, x):
        '''
        x: shape (n, t, c, h, w)
        '''
        assert len(x.size()) == 5
        n, t, c, h, w = x.size()
        x = self.module(x)
        return x

if __name__ == '__main__':
    a = attention_A(16, 15)
    data = torch.zeros(3, 3, 16, 224, 224)
    out = a(data)
    print(out.shape)