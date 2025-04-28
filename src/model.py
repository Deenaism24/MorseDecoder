import torch
import torch.nn as nn
import math
from typing import List

from src import config


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), pool_kernel=(2,2), pool_stride=(2,2)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class MorseCRNN(nn.Module):
    def __init__(self,
                 num_classes: int,          # Количество классов = размер словаря + 1 для blank
                 n_mels: int = config.N_MELS,  # Высота входной спектрограммы = количество Mel-фильтров
                 cnn_out_channels: List[int] = [32, 64, 128, 128],  # Каналы в CNN блоках
                 rnn_hidden_size: int = config.RNN_HIDDEN_SIZE,
                 rnn_num_layers: int = config.RNN_NUM_LAYERS,
                 rnn_bidirectional: bool = config.RNN_BIDIRECTIONAL,
                 rnn_dropout: float = config.RNN_DROPOUT,
                 rnn_type: str = 'LSTM'): # 'LSTM' или 'GRU'
        super().__init__()

        self.num_classes = num_classes
        self._n_mels = n_mels

        # --- Сверточная часть (CNN) ---
        cnn_layers = []
        cur_in_channels = 1

        pool_kernels = [(2, 2), (2, 2), (2, 1), (2, 1)]
        pool_strides = pool_kernels

        if len(pool_kernels) != len(cnn_out_channels):
            raise ValueError("Length of pool_kernels must match length of cnn_out_channels")

        for i, out_ch in enumerate(cnn_out_channels):
            cnn_layers.append(CNNBlock(
                in_channels=cur_in_channels,
                out_channels=out_ch,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                pool_kernel=pool_kernels[i],
                pool_stride=pool_strides[i]
            ))
            cur_in_channels = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # (batch, channels, height, width) -> (batch, seq_len, features)
        self.cnn_out_height = self._calculate_out_dim(n_mels, [p[0] for p in pool_kernels])
        self.cnn_out_width_reduction = 2**sum(p[1] // p[0] for p in pool_kernels if p[1] > 0 and p[0] > 0)

        in_size = cur_in_channels * self.cnn_out_height
        print(f"CNN output height: {self.cnn_out_height}, Channels: {cur_in_channels}")
        print(f"RNN input size (cnn_channels * cnn_height): {in_size}")
        if in_size == 0:
             raise ValueError("Calculated RNN input size is 0. Check CNN architecture and input n_mels.")

        # --- Рекуррентная часть (RNN) ---
        rnn_class = nn.LSTM if rnn_type.upper() == 'LSTM' else nn.GRU
        self.rnn = rnn_class(
            input_size=in_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=rnn_bidirectional,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0,
            batch_first=True
        )

        # --- Линейный выходной слой ---
        rnn_out_features = rnn_hidden_size * (2 if rnn_bidirectional else 1)
        self.fc = nn.Linear(rnn_out_features, num_classes)

        self.log_softmax = nn.LogSoftmax(dim=2)

    def _calculate_out_dim(self, in_dim, pool_kernels):
        dim = in_dim
        for k in pool_kernels:
            dim = math.floor(dim / k)
        return dim

    def get_out_seq_len(self, in_seq_len: torch.Tensor) -> torch.Tensor:
        pool_time_kernels = [p[1] for p in [(2, 2), (2, 2), (2, 1), (2, 1)]]

        out_len = in_seq_len.float()
        for k in pool_time_kernels:
            if k > 1:
                out_len = torch.floor(out_len / k)

        return out_len.long()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)

        # 1. Проход через CNN
        x = self.cnn(x)

        # 2. Решейп для RNN
        N, C_out, H_out, W_out = x.shape
        x = x.permute(0, 3, 1, 2)
        in_size = C_out * H_out
        x = x.reshape(N, W_out, in_size)

        # 3. Проход через RNN
        x, _ = self.rnn(x)

        # 4. Проход через выходной линейный слой
        x = self.fc(x)

        # 5. Применение LogSoftmax
        x = self.log_softmax(x)

        # 6. Перестановка для CTC Loss
        x = x.permute(1, 0, 2)
        return x
