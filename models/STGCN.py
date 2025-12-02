import torch
import torch.nn as nn
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .base_model import BaseModel

class STConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STConvBlock, self).__init__()
        self.temporal_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.graph_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))
        self.temporal_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.layer_norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = x.permute(0, 2, 1, 3)
        residual = self.temporal_conv1(x)

        if adj.dim() == 2:
            adj = adj.unsqueeze(0)

        x_gconv = torch.einsum('bcnl,bnm->bcml', residual, adj)

        x_gconv = self.relu(self.graph_conv(x_gconv))
        x_out = self.temporal_conv2(x_gconv)
        out_sum = residual + x_out

        out_permuted = out_sum.permute(0, 2, 3, 1).contiguous()
        out_normed = self.layer_norm(out_permuted)
        return self.relu(out_normed).permute(0, 1, 3, 2).contiguous()


class STGCN(BaseModel):
    def __init__(self, config, data_module):
        super().__init__(config) 

        model_config = config['model']
        self.hidden_channels = model_config['hidden_channels']

        self.st_conv1 = STConvBlock(data_module.in_channels, self.hidden_channels)
        self.st_conv2 = STConvBlock(self.hidden_channels, self.hidden_channels)
        self.final_conv = nn.Conv2d(in_channels=self.hidden_channels, out_channels=1, kernel_size=(1, 1))

        self.fully_connected = nn.Linear(data_module.num_timesteps_input, data_module.num_timesteps_output)

    def forward(self, x, adj):
        x = self.st_conv1(x, adj)
        x = self.st_conv2(x, adj)
        x = x.permute(0, 2, 1, 3)
        x = self.final_conv(x)
        x = x.squeeze(1)
        x = self.fully_connected(x)
        return x.unsqueeze(2)