import torch.nn as nn
from .gelu import GELU
from einops import rearrange


class PositionwiseFeedForward(nn.Module):
    # Implements FFN equation with dropout regularization.
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv2d(in_channels=d_model[1], out_channels=d_ff, kernel_size=3, stride=1, padding=1)
        self.w_2 = nn.Conv2d(in_channels=d_ff, out_channels=d_model[1], kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(d_ff)
        self.conv2_bn = nn.BatchNorm2d(d_model[1])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        b, n, h, w, l = x.shape
        x = rearrange(x, 'b n h w l -> (b l) n h w', b=b, n=n, h=h, w=w, l=l)
        x = self.w_1(x)
        x = self.conv1_bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.conv2_bn(x)
        return rearrange(x, '(b l) n h w -> b n h w l', l=l)