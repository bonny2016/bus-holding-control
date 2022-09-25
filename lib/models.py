from cProfile import run
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import self_attention

class simpleFF(nn.Module):
    def __init__(self, shape, actions_n):
        super(simpleFF, self).__init__()
        state_steps = shape[0]
        size = int(np.prod(shape))
        self.ff = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, actions_n)
        )

    def forward(self, x):
        arr = x.view(x.size()[0], -1)
        # print("x.shape[0]:", x.shape[0])
        result = self.ff(arr)
        return result

class DQNConvNStepBusBased(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConvNStepBusBased, self).__init__()
        channel = shape[0]

        self.conv = nn.Sequential(
            nn.Conv2d(channel, 8, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(3,3)),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)
        print(f"out_size:{out_size}")

        self.ff = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, actions_n)
        )

    def _get_conv_out(self, shape):
        tmp = torch.zeros(1, *shape)
        o = self.conv(tmp)
        return int(np.prod(o.size()))

    def forward(self, x):
        # x = torch.from_numpy(x).float()
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size()[0], -1)
        result = self.ff(conv_out)
        return result        
        
class TransformDQN(nn.Module):
    def __init__(self, shape, actions_n = 2, device="cpu", forward_expandsion=2, dropout=0.1):
        super(TransformDQN, self).__init__()
        steps, heads, head_dim = shape
        self.device = device
        self.encoder = self_attention.Encoder(head_dim, heads, 1, device,forward_expandsion)
        out_size = self._get_encode_out(shape)
        print(f"encoder_out_size:{out_size}")
        
        self.q_val = nn.Sequential(
            nn.Linear(out_size, 64),
            nn.ReLU(),
            nn.Linear(64, actions_n)
        )

    def _get_encode_out(self, shape):
        #x = torch.zeros(1, *shape)
        x = torch.from_numpy(np.zeros([1, *shape])).float().to(self.device)
        encoded = self.encoder(x)
        return int(np.prod(encoded.size()))    

    def forward(self, x):
        x = x.to(self.device)
        encoded = self.encoder(x).view(x.size()[0], -1)
        #print(f"shape after encode:{encoded.shape}")
        return self.q_val(encoded)
