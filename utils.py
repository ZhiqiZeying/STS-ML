import torch
import torch.nn.functional as F
import config
from torch.utils.data import Dataset

device = config.device
batch_size = config.batch_size


def adj_matrix_to_edge_index(adj_matrix, sparse=True):
    # 假设adj_matrix是一个二维的PyTorch张量
    # 首先找到非零元素的索引（即边的索引）
    row, col = torch.nonzero(adj_matrix, as_tuple=True)

    # 对于无权图，我们简单地创建一个全1的权重张量
    weight = torch.ones_like(row, dtype=torch.float)
    for i in range(len(row)):
        weight[i] = adj_matrix[row[i], col[i]]

    # 如果需要稀疏表示，则直接返回edge_index和weight
    if sparse:
        return row, col, weight

    # 如果需要密集表示（例如COO格式的稀疏张量），则创建一个COO张量
    # 注意：这里我们并不真正创建一个COO张量，只是展示如何组织数据
    # edge_index = torch.stack([row, col], dim=0)
    # return edge_index, weight


def sub_graph(adj_matrix, where):

    # 创建一个布尔掩码来选择这些节点对应的行和列
    row_mask = torch.zeros((adj_matrix.size(0), 1), dtype=torch.int)
    row_mask[where] = 1
    col_mask = row_mask.clone().squeeze(1).unsqueeze(0)  # 列掩码与行掩码相同
    mask = row_mask@col_mask
    mask = mask.to(device)

    # 使用掩码从邻接矩阵中提取子矩阵
    # sub_adj_matrix = adj_matrix[row_mask][:, col_mask]
    sub_adj_matrix = adj_matrix*mask
    row, col, weight = adj_matrix_to_edge_index(sub_adj_matrix)
    return row, col, weight


class eegDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor.to(device)
        self.y = y_tensor.to(device)
        # self.g1 = x_tensor  # Pre-Frontal
        # self.g1[:, :5, :] = 0
        # self.g2 = x_tensor  # Left Frontal
        # self.g2[:, 5:8, :] = 0
        # self.g2[:, 14:17, :] = 0
        # self.g3 = x_tensor  # Frontal
        # self.g3[:, 8:11, :] = 0
        # self.g3[:, 17:20, :] = 0
        # self.g4 = x_tensor  # Right Frontal
        # self.g4[:, 11:14, :] = 0
        # self.g4[:, 20:23, :] = 0
        # self.g5 = x_tensor  # Left Temporal
        # self.g5[:, 23:26, :] = 0
        # self.g5[:, 32:35, :] = 0
        # self.g6 = x_tensor  # Central
        # self.g6[:, 26:29, :] = 0
        # self.g6[:, 35:38, :] = 0
        # self.g7 = x_tensor  # Right Temporal
        # self.g7[:, 29:32, :] = 0
        # self.g7[:, 38:41, :] = 0
        # self.g8 = x_tensor  # Left Parietal
        # self.g8[:, 41:44, :] = 0
        # self.g8[:, 50:52, :] = 0
        # self.g8[:, 57:58, :] = 0
        # self.g9 = x_tensor  # Occipital
        # self.g9[:, 44:47, :] = 0
        # self.g9[:, 52:55, :] = 0
        # self.g9[:, 58:61, :] = 0
        # self.g10 = x_tensor  # Right parietal
        # self.g10[:, 47:50, :] = 0
        # self.g10[:, 55:57, :] = 0
        # self.g10[:, 61:62, :] = 0

        # self.f1 = x_tensor
        # self.f1[:, :, 0] = 0
        # self.f2 = x_tensor
        # self.f2[:, :, 1] = 0
        # self.f3 = x_tensor
        # self.f3[:, :, 2] = 0
        # self.f4 = x_tensor
        # self.f4[:, :, 3] = 0
        # self.f5 = x_tensor
        # self.f5[:, :, 4] = 0

    def __getitem__(self, index):
        # , self.g1[index], self.g2[index], self.g3[index], self.g4[index], self.g5[index], self.g6[index], self.g7[index], self.g8[index], self.g9[index], self.g10[index], self.f1[index], self.f2[index], self.f3[index], self.f4[index], self.f5[index]
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


def calc_diff_loss(outs, margin=1.0, regularization_weight=0.01):
    diff_loss = 0.0
    for i in range(len(outs)):
        for j in range(i+1, len(outs)):
            for k in range(batch_size):
                # 计算两个嵌入表示之间的距离（欧氏距离的平方）
                diff = torch.norm(outs[i][k, :] - outs[j]
                                  [k, :], dim=0, p=2) ** 2
                # 使用ReLU确保只有差异小于margin的才会产生损失
                diff_loss += torch.relu(margin - diff)

    # 正则化项
    reg_loss = regularization_weight * \
        sum(out.norm(dim=0).mean() for out in outs)

    # 总损失 = 差异最大化损失项 + 正则化项
    total_loss = diff_loss / \
        (len(outs) * (len(outs) - 1) // 2) + reg_loss

    return total_loss


def generate_cheby_adj(A, K, device):
    support = []
    for i in range(K):
        if i == 0:
            # support.append(torch.eye(A.shape[1]).cuda())  #torch.eye生成单位矩阵
            temp = torch.eye(A.shape[1])
            temp = temp.to(device)
            support.append(temp)
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support

# adj_matrix = torch.tensor([[0, 3, 0, 2],
#                            [1, 0, 5, 0],
#                            [0, 4, 0, 8],
#                            [9, 0, 6, 0]], dtype=torch.float)
# row, col, weight = sub_graph(adj_matrix, [i for i in range(2)]+[3])
# print("Edge indices:")
# print(row)
# print(col)
# print("Edge weights:")
# print(weight)
