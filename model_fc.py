# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from torch import nn
import torch
import math

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 512)
        self.lrelu = nn.LeakyReLU(0.2, False)
        self.dropout = nn.Dropout(p=0.5, inplace=True)

    def forward(self, x):
        input_x = x
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        out = input_x + x
        return out
    
    
class ActionText(nn.Module):
    def __init__(self):
        super().__init__()
        res_blocks = []
        for _ in range(3):
            res_blocks.append(ResBlock())
        self.res_blocks = nn.Sequential(*res_blocks)
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = self.res_blocks(x)
        out = self.fc(x)
        return out
