import torch

epochs = 20
batch_size = 32
lr_list = [[1, 1e-3, 1]]
weight_decay = 1e-4
drop_rate = 0.25
num_workers = 0
device = ('cuda:2' if torch.cuda.is_available() else 'cpu')
K = 2
