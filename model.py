import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from utils import adj_matrix_to_edge_index, sub_graph, generate_cheby_adj
import torch.nn.functional as F
import numpy as np
import config

from layers import *

device = config.device


class Chebynet(nn.Module):
    def __init__(self, inchannel, K, outchannel, xdim):
        super(Chebynet, self).__init__()
        self.K = K
        self.outchannel = outchannel
        self.xdim = xdim
        self.gc1 = nn.ModuleList()  # https://zhuanlan.zhihu.com/p/75206669
        for i in range(K):
            self.gc1.append(GraphConvolution(inchannel, outchannel))
        # self.conv = nn.Conv1d(outchannel, inchannel, 3, padding='same')

    def forward(self, x, L):
        device = x.device
        adj = generate_cheby_adj(L, self.K, device)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class DE_MoE(nn.Module):
    def __init__(self, inchannel, outchannel, xdim, linearsize=512, dropout=0.0, testmode=False, **kwargs):
        super(DE_MoE, self).__init__()
        self.outchannel = outchannel
        # self.batch = batch
        self.xdim = xdim
        self.testmode = testmode
        self.c = int(kwargs['HC'])
        # self.bn = nn.BatchNorm2d(xdim)
        self.conv = Chebynet(inchannel, config.K, outchannel, xdim)
        self.unet = UNet(inchannel, outchannel, bilinear=False)
        self.conv2d = nn.Conv2d(xdim, xdim, (10, 1), groups=xdim)
        # self.conv2d = nn.Conv2d(xdim, xdim, (10, 1))
        self.HC = nn.Sequential(
            nn.Linear(outchannel * 62 * 26, linearsize),
            nn.BatchNorm1d(linearsize),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(linearsize, linearsize // 2),
            nn.BatchNorm1d(linearsize // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(linearsize // 2, kwargs['HC'])
        )
        self.A = nn.Parameter(torch.FloatTensor(xdim, xdim))  # 领接矩阵
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(10.0))
        self.c = nn.Parameter(torch.ones(1, 15), requires_grad=True)
        self.d = nn.Parameter(torch.ones(10))
        self.e = nn.Parameter(torch.ones(5) * 2)
        self.num = int(outchannel/inchannel)
        nn.init.xavier_normal_(self.A)

    def forward(self, *args):
        x = args[0]

        # x = self.bn(x)

        out1 = x.clone()
        out1 = out1.permute(0, 3, 1, 2)
        out1 = self.unet(out1)
        out1 = out1.permute(0, 2, 3, 1)

        # if not self.testmode:
        #     for i in range(x.size(0)):  # 遍历第一个维度（大小为4）
        #         for j in range(x.size(2)):  # 遍历第三个维度（大小为10）
        #             # 为当前切片生成新的噪声并加上
        #             noise = torch.randn(62, 5).to(device)  # 为每个切片生成新的高斯噪声
        #             x[i, :, j, :] += noise

        sigma =torch.var(x, dim=2, keepdim=True)
        x = self.conv2d(x) * torch.exp(-sigma)
        x = x.squeeze(2)

        # g1 = args[1]
        # g2 = args[2]
        # g3 = args[3]
        # g4 = args[4]
        # g5 = args[5]
        # g6 = args[6]
        # g7 = args[7]
        # g8 = args[8]
        # g9 = args[9]
        # g10 = args[10]
        # f1 = args[11]
        # f2 = args[11]
        # f3 = args[11]
        # f4 = args[11]
        # f5 = args[11]

        # # 遍历每个样本
        # for i in range(x.size(0)):
        #     # 为当前样本生成6个不重复的随机索引
        #     indices = torch.randperm(62)[:6]

        #     # 创建一个二维的mask，其中选中的通道为False（0），其余为True（1）
        #     # 但由于我们要将选中的通道设置为0，所以实际上我们想要的是反向的mask
        #     mask = torch.ones(62, dtype=torch.bool)
        #     mask[indices] = False  # 将选中的索引位置设置为False

        #     # 将mask扩展到与x_modified[i]相同的形状
        #     mask_expanded = mask.unsqueeze(1).expand_as(x[i])

        #     # 应用mask，将选中的通道设置为0
        #     x[i][mask_expanded] = 0

        g1 = x.clone()  # Pre-Frontal
        g1[:, :5, :] = 0
        g2 = x.clone()  # Left Frontal
        g2[:, 5:8, :] = 0
        g2[:, 14:17, :] = 0
        g3 = x.clone()  # Frontal
        g3[:, 8:11, :] = 0
        g3[:, 17:20, :] = 0
        g4 = x.clone()  # Right Frontal
        g4[:, 11:14, :] = 0
        g4[:, 20:23, :] = 0
        g5 = x.clone()  # Left Temporal
        g5[:, 23:26, :] = 0
        g5[:, 32:35, :] = 0
        g6 = x.clone()  # Central
        g6[:, 26:29, :] = 0
        g6[:, 35:38, :] = 0
        g7 = x.clone()  # Right Temporal
        g7[:, 29:32, :] = 0
        g7[:, 38:41, :] = 0
        g8 = x.clone()  # Left Parietal
        g8[:, 41:44, :] = 0
        g8[:, 50:52, :] = 0
        g8[:, 57:58, :] = 0
        g9 = x.clone()  # Occipital
        g9[:, 44:47, :] = 0
        g9[:, 52:55, :] = 0
        g9[:, 58:61, :] = 0
        g10 = x.clone()  # Right parietal
        g10[:, 47:50, :] = 0
        g10[:, 55:57, :] = 0
        g10[:, 61:62, :] = 0

        f1 = x.clone()
        f1[:, :, 0] = 0
        f2 = x.clone()
        f2[:, :, 1] = 0
        f3 = x.clone()
        f3[:, :, 2] = 0
        f4 = x.clone()
        f4[:, :, 3] = 0
        f5 = x.clone()
        f5[:, :, 4] = 0

        g1 = F.relu(self.conv(g1, self.A))
        g2 = F.relu(self.conv(g2, self.A))
        g3 = F.relu(self.conv(g3, self.A))
        g4 = F.relu(self.conv(g4, self.A))
        g5 = F.relu(self.conv(g5, self.A))
        g6 = F.relu(self.conv(g6, self.A))
        g7 = F.relu(self.conv(g7, self.A))
        g8 = F.relu(self.conv(g8, self.A))
        g9 = F.relu(self.conv(g9, self.A))
        g10 = F.relu(self.conv(g10, self.A))

        f1 = F.relu(self.conv(f1, self.A))
        f2 = F.relu(self.conv(f2, self.A))
        f3 = F.relu(self.conv(f3, self.A))
        f4 = F.relu(self.conv(f4, self.A))
        f5 = F.relu(self.conv(f5, self.A))

        out2 = F.relu(self.conv(x, self.A))

        # out3 = x.clone()
        # out3 = out3.repeat(1, 1, self.num)

        # +self.d * (g1+g2+g3+g4+g5+g6+g7+g8+g9+g10)+self.e*(f1+f2+f3+f4+f5)
        g = [g1,g2,g3,g4,g5,g6,g7,g8,g9,g10]
        b1 = [a * b for a, b in zip(self.d, g)]
        B1 = torch.stack(b1, dim=2)
        f = [f1,f2,f3,f4,f5]
        b2 = [a * b for a, b in zip(self.e, f)]
        B2 = torch.stack(b2, dim=2)
        out = torch.stack([out2,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,f1,f2,f3,f4,f5],dim=2)
        out = torch.cat([out, self.a*out1], dim=2)
        out3 = torch.cat([out2.unsqueeze(2),B1,B2], dim=2)
        out3 = torch.cat([out3, self.a*out1], dim=2)
        # out3 = torch.cat([self.b*out2.unsqueeze(2), self.d * (g1+g2+g3+g4+g5+g6+g7+g8+g9+g10).unsqueeze(2)], dim=2)
        # out3 = torch.cat([out3, self.e*(f1+f2+f3+f4+f5).unsqueeze(2)], dim=2)
        # # out = out.unsqueeze(2)
        # out3 = torch.cat([out3, self.a*out1], dim=2)
        # out = torch.cat([out2.unsqueeze(2), (g1+g2+g3+g4+g5+g6+g7+g8+g9+g10).unsqueeze(2)], dim=2)
        # out = torch.cat([out, (f1+f2+f3+f4+f5).unsqueeze(2)], dim=2)
        # # out = out.unsqueeze(2)
        # out = torch.cat([out, self.a*out1], dim=2)
        out = F.softmax(out3, dim=2) * out

        out = out.reshape(x.shape[0], -1)
        out = self.HC(out)
        out = F.softmax(out, dim=1)
        return out

        # if not self.testmode:
        #     # x1 = x[:5]  # Pre-Frontal
        #     # x2 = torch.vstack((x[5:8], x[14:17]))  # Left Frontal
        #     # x3 = torch.vstack((x[8:11], x[17:20]))  # Frontal
        #     # x4 = torch.vstack((x[11:14], x[20:23]))  # Right Frontal
        #     # x5 = torch.vstack((x[23:26], x[32:35]))  # Left Temporal
        #     # x6 = torch.vstack((x[26:29], x[35:38]))  # Central
        #     # x7 = torch.vstack((x[29:32], x[38:41]))  # Right Temporal
        #     # x8 = torch.vstack(
        #     #     (torch.vstack((x[41:44], x[50:52])), x[57]))  # Left Parietal
        #     # x9 = torch.vstack(
        #     #     (torch.vstack((x[44:47], x[52:55])), x[58:61]))  # Occipital
        #     # x10 = torch.vstack(
        #     #     (torch.vstack((x[47:50], x[55:57])), x[61]))  # Right parietal

        #     # x11 = torch.vstack((torch.vstack((torch.vstack((torch.vstack((torch.vstack((torch.vstack((torch.vstack((torch.vstack((torch.vstack(
        #     #     # Left hemisphere
        #     #     (x[0], x[1])), x[3])), x[5:10])), x[14:19])), x[23:28])), x[32:37])), x[41:46])), x[50:54])), x[57:60]))
        #     # x12 = torch.vstack((torch.vstack((torch.vstack((torch.vstack((torch.vstack((torch.vstack((torch.vstack((torch.vstack((torch.vstack(
        #     #     # Left hemisphere
        #     #     (x[0], x[2])), x[4])), x[9:14])), x[18:23])), x[27:32])), x[36:41])), x[45:50])), x[53:57])), x[59:62]))

        #     # x = x  # Whole brain

        #     # row1, col1, w1 = sub_graph(self.A, [i for i in range(5)])
        #     # e1 = torch.vstack((row1, col1)).long()
        #     # row2, col2, w2 = sub_graph(
        #     #     self.A, [i+5 for i in range(3)]+[i+14 for i in range(3)])
        #     # e2 = torch.vstack((row2, col2)).long()
        #     # row3, col3, w3 = sub_graph(
        #     #     self.A, [i+8 for i in range(3)]+[i+17 for i in range(3)])
        #     # e3 = torch.vstack((row3, col3)).long()
        #     # row4, col4, w4 = sub_graph(
        #     #     self.A, [i+11 for i in range(3)]+[i+20 for i in range(3)])
        #     # e4 = torch.vstack((row4, col4)).long()
        #     # row5, col5, w5 = sub_graph(
        #     #     self.A, [i+23 for i in range(3)]+[i+32 for i in range(3)])
        #     # e5 = torch.vstack((row5, col5)).long()
        #     # row6, col6, w6 = sub_graph(
        #     #     self.A, [i+26 for i in range(3)]+[i+35 for i in range(3)])
        #     # e6 = torch.vstack((row6, col6)).long()
        #     # row7, col7, w7 = sub_graph(
        #     #     self.A, [i+29 for i in range(3)]+[i+38 for i in range(3)])
        #     # e7 = torch.vstack((row7, col7)).long()
        #     # row8, col8, w8 = sub_graph(
        #     #     self.A, [i+41 for i in range(3)]+[i+50 for i in range(2)]+[57])
        #     # e8 = torch.vstack((row8, col8)).long()
        #     # row9, col9, w9 = sub_graph(
        #     #     self.A, [i+44 for i in range(3)]+[i+52 for i in range(3)]+[i+58 for i in range(3)])
        #     # e9 = torch.vstack((row9, col9)).long()
        #     # row10, col10, w10 = sub_graph(
        #     #     self.A, [i+47 for i in range(3)]+[i+55 for i in range(2)]+[61])
        #     # e10 = torch.vstack((row10, col10)).long()

        #     # row11, col11, w11 = sub_graph(
        #     #     self.A, [0]+[1]+[3]+[i+5 for i in range(5)]+[i+14 for i in range(5)]+[i+23 for i in range(5)]+[i+32 for i in range(5)]+[i+41 for i in range(5)]+[i+50 for i in range(4)]+[i+57 for i in range(3)])
        #     # e11 = torch.vstack((row11, col11)).long()
        #     # row12, col12, w12 = sub_graph(
        #     #     self.A, [0]+[2]+[4]+[i+9 for i in range(5)]+[i+18 for i in range(5)]+[i+27 for i in range(5)]+[i+36 for i in range(5)]+[i+45 for i in range(5)]+[i+53 for i in range(4)]+[i+59 for i in range(3)])
        #     # e12 = torch.vstack((row12, col12)).long()

        #     # row, col, w = adj_matrix_to_edge_index(self.A)
        #     # e = torch.vstack((row, col)).long()

        #     # # GCN
        #     # x1 = F.relu(self.conv1(x, e1, w1.abs()))
        #     # x2 = F.relu(self.conv1(x, e2, w2.abs()))
        #     # x3 = F.relu(self.conv1(x, e3, w3.abs()))
        #     # x4 = F.relu(self.conv1(x, e4, w4.abs()))
        #     # x5 = F.relu(self.conv1(x, e5, w5.abs()))
        #     # x6 = F.relu(self.conv1(x, e6, w6.abs()))
        #     # x7 = F.relu(self.conv1(x, e7, w7.abs()))
        #     # x8 = F.relu(self.conv1(x, e8, w8.abs()))
        #     # x9 = F.relu(self.conv1(x, e9, w9.abs()))
        #     # x10 = F.relu(self.conv1(x, e10, w10.abs()))

        #     # x11 = F.relu(self.conv1(x, e11, w11.abs()))
        #     # x12 = F.relu(self.conv1(x, e12, w12.abs()))

        #     out2 = F.relu(self.conv(x, self.A))
        #     out = self.a * out + self.b * x

        #     # g1 = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10
        #     # g2 = x11 + x12
        #     # g3 = x
        #     # out = self.a*g1 + self.b*g2 + self.c*g3

        #     # out = F.relu(self.conv2(g, e, w.abs()))

        #     # x1 = x1.view(self.batch, -1)
        #     # x2 = x2.view(self.batch, -1)
        #     # x3 = x3.view(self.batch, -1)
        #     # x4 = x4.view(self.batch, -1)
        #     # x5 = x5.view(self.batch, -1)
        #     # x6 = x6.view(self.batch, -1)
        #     # x7 = x7.view(self.batch, -1)
        #     # x8 = x8.view(self.batch, -1)
        #     # x9 = x9.view(self.batch, -1)
        #     # x10 = x10.view(self.batch, -1)
        #     # x11 = x11.view(self.batch, -1)
        #     # x12 = x12.view(self.batch, -1)
        #     out = out.view(self.batch, -1)

        #     # x1 = self.Projection(x1)
        #     # x2 = self.Projection(x2)
        #     # x3 = self.Projection(x3)
        #     # x4 = self.Projection(x4)
        #     # x5 = self.Projection(x5)
        #     # x6 = self.Projection(x6)
        #     # x7 = self.Projection(x7)
        #     # x8 = self.Projection(x8)
        #     # x9 = self.Projection(x9)
        #     # x10 = self.Projection(x10)
        #     # x11 = self.Projection(x11)
        #     # x12 = self.Projection(x12)
        #     out = self.HC(out)

        #     # x1 = F.normalize(x1, dim=-1)
        #     # x2 = F.normalize(x2, dim=-1)
        #     # x3 = F.normalize(x3, dim=-1)
        #     # x4 = F.normalize(x4, dim=-1)
        #     # x5 = F.normalize(x5, dim=-1)
        #     # x6 = F.normalize(x6, dim=-1)
        #     # x7 = F.normalize(x7, dim=-1)
        #     # x8 = F.normalize(x8, dim=-1)
        #     # x9 = F.normalize(x9, dim=-1)
        #     # x10 = F.normalize(x10, dim=-1)
        #     # x11 = F.normalize(x11, dim=-1)
        #     # x12 = F.normalize(x12, dim=-1)
        #     out = F.softmax(out, dim=1)
        #     # return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, out
        #     return out
        # else:
        #     # row, col, w = adj_matrix_to_edge_index(self.A)
        #     # e = torch.vstack((row, col)).long()
        #     out2 = F.relu(self.conv(x, self.A))
        #     out = self.a * out + self.b * x
        #     # out = F.relu(self.conv2(x, e, w))
        #     out = out.view(self.batch, -1)
        #     out = self.HC(out)
        #     out = F.softmax(out, dim=1)
        #     return out



class SPDProjection(nn.Module):
    def forward(self, x):
        B, C, F = x.shape
        mu = x.mean(dim=1, keepdim=True)
        x_centered = x - mu
        cov = torch.matmul(x_centered, x_centered.transpose(1,2)) / F
        cov = cov + 1e-6 * torch.eye(C, device=x.device)
        return cov



class DualOrthogonalAttention(nn.Module):
    def __init__(self, dim=65, nheads=10, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.nheads = nheads
        self.Wq1 = nn.Linear(dim, dim*nheads)
        self.Wk1 = nn.Linear(dim, dim*nheads)
        self.Wq2 = nn.Linear(dim, dim*nheads) 
        self.Wk2 = nn.Linear(dim, dim*nheads)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, Fn, Fr):
        bs, node, dim1 = Fn.shape  # (bs,62,65)
        bs, region, dim2 = Fr.shape  # (bs,10,65)
        # 区域到节点注意力
        Q_r2n = self.Wq1(Fn).view(bs, node, self.nheads, self.dim).transpose(1, 2)  # (bs,10,62,65)
        K_r2n = self.Wk1(Fr).view(bs, region, self.nheads, self.dim).transpose(1, 2)  # (bs,10,10,65)
        A_r2n = torch.softmax(torch.matmul(Q_r2n, K_r2n.transpose(-2, -1)) / (self.dim ** 0.5), dim=-1)  # (bs,10,62,10)
        
        # 节点到区域注意力  
        Q_n2r = self.Wq2(Fr).view(bs, region, self.nheads, self.dim).transpose(1, 2)  # (bs,10,10,65)
        K_n2r = self.Wk2(Fn).view(bs, node, self.nheads, self.dim).transpose(1, 2)  # (bs,10,62,65)
        A_n2r = torch.softmax(torch.matmul(Q_n2r, K_n2r.transpose(-2, -1)) / (self.dim ** 0.5), dim=-1)  # (bs,10,10,62)
        
        # 正交约束损失
        orth_loss = torch.norm(A_r2n @ A_n2r.transpose(-2, -1) - torch.eye(62), p='fro', dim=(-2, -1))
        orth_loss = orth_loss.mean()
        
        # 动态门控融合
        g = self.gate(torch.cat([Fn.mean(1), Fr.mean(1)], dim=-1))  # (bs,65)
        fused = torch.cat([g * (A_r2n @ Fr), (1-g) * (A_n2r @ Fn)], dim=2)
        return fused, orth_loss



class DRHG_MoE(nn.Module):
    def __init__(self, inchannel, midchannel, outchannel, xdim, k1, linearsize=512, dropout=0.0, testmode=False, **kwargs):
        super(DRHG_MoE, self).__init__()
        self.outchannel = outchannel
        self.xdim = xdim
        self.testmode = testmode
        self.Cheby1 = Chebynet(inchannel, config.K, outchannel, xdim)
        self.Region = nn.ModuleDict({
            "conv1": nn.Conv1d(in_channels=inchannel, out_channels=midchannel, kernel_size=5),
            "conv2": nn.Conv1d(in_channels=inchannel, out_channels=midchannel, kernel_size=6),
            "conv3": nn.Conv1d(in_channels=inchannel, out_channels=midchannel, kernel_size=6),
            "conv4": nn.Conv1d(in_channels=inchannel, out_channels=midchannel, kernel_size=6),
            "conv5": nn.Conv1d(in_channels=inchannel, out_channels=midchannel, kernel_size=6),
            "conv6": nn.Conv1d(in_channels=inchannel, out_channels=midchannel, kernel_size=6),
            "conv7": nn.Conv1d(in_channels=inchannel, out_channels=midchannel, kernel_size=6),
            "conv8": nn.Conv1d(in_channels=inchannel, out_channels=midchannel, kernel_size=6),
            "conv9": nn.Conv1d(in_channels=inchannel, out_channels=midchannel, kernel_size=9),
            "conv10": nn.Conv1d(in_channels=inchannel, out_channels=midchannel, kernel_size=6)
        })
        self.Cheby2 = Chebynet(midchannel, config.K, outchannel, k1)
        # self.unet = UNet(inchannel, outchannel, bilinear=False)
        self.TFE = nn.Conv2d(xdim, xdim, (10, 1), groups=xdim)
        self.HC = nn.Sequential(
            nn.Linear(outchannel * 72, linearsize),
            nn.BatchNorm1d(linearsize),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(linearsize, linearsize // 2),
            nn.BatchNorm1d(linearsize // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(linearsize // 2, kwargs['HC'])
        )
        self.A = nn.Parameter(torch.FloatTensor(xdim, xdim))  # 整图领接矩阵
        self.A1 = nn.Parameter(torch.FloatTensor(k1, k1))  # 区域领接矩阵
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))
        self.c = nn.Parameter(torch.tensor(1.0))
        # self.d = nn.Parameter(torch.ones(10))
        # self.e = nn.Parameter(torch.ones(5) * 2)
        self.num = int(outchannel/inchannel)
        nn.init.xavier_normal_(self.A)
        nn.init.xavier_normal_(self.A1)
        
    def forward(self, *args):
        x = args[0]

        # out1 = x.clone()
        # out1 = out1.permute(0, 3, 1, 2)
        # out1 = self.unet(out1)
        # out1 = out1.permute(0, 2, 3, 1)

        sigma =torch.var(x, dim=2, keepdim=True)
        g = self.TFE(x) * torch.exp(-sigma)
        g = g.squeeze(2)
        
        g1 = F.relu(self.Cheby1(g, self.A))
        
        g21 = g[:,:5,:].clone()  # Pre-Frontal
        g22 = torch.cat([g[:,5:8,:].clone(), g[:,14:17,:].clone()], dim=1)  # Left Frontal
        g23 = torch.cat([g[:,8:11,:].clone(), g[:,17:20,:].clone()], dim=1)  # Frontal
        g24 = torch.cat([g[:,11:14,:].clone(), g[:,20:23,:].clone()], dim=1)  # Right Frontal
        g25 = torch.cat([g[:,23:26,:].clone(), g[:,32:35,:].clone()], dim=1)  # Left Temporal
        g26 = torch.cat([g[:,26:29,:].clone(), g[:,35:38,:].clone()], dim=1)  # Central
        g27 = torch.cat([g[:,29:32,:].clone(), g[:,38:41,:].clone()], dim=1)  # Right Temporal
        g28 = torch.cat(
            [torch.cat([g[:,41:44,:].clone(), g[:,50:52,:].clone()], dim=1), g[:,57:58,:].clone()], dim=1)  # Left Parietal
        g29 = torch.cat(
            [torch.cat([g[:,44:47,:].clone(), g[:,52:55,:].clone()], dim=1), g[:,58:61,:].clone()], dim=1)  # Occipital
        g30 = torch.cat(
            [torch.cat([g[:,47:50,:].clone(), g[:,55:57,:].clone()], dim=1), g[:,61:62,:].clone()], dim=1)  # Right parietal
        
        RG = [
            self.Region["conv1"](g21.permute(0, 2, 1)).permute(0, 2, 1),
            self.Region["conv2"](g22.permute(0, 2, 1)).permute(0, 2, 1),
            self.Region["conv3"](g23.permute(0, 2, 1)).permute(0, 2, 1),
            self.Region["conv4"](g24.permute(0, 2, 1)).permute(0, 2, 1),
            self.Region["conv5"](g25.permute(0, 2, 1)).permute(0, 2, 1),
            self.Region["conv6"](g26.permute(0, 2, 1)).permute(0, 2, 1),
            self.Region["conv7"](g27.permute(0, 2, 1)).permute(0, 2, 1),
            self.Region["conv8"](g28.permute(0, 2, 1)).permute(0, 2, 1),
            self.Region["conv9"](g29.permute(0, 2, 1)).permute(0, 2, 1),
            self.Region["conv10"](g30.permute(0, 2, 1)).permute(0, 2, 1)
        ]
        
        g2 = torch.cat(RG, dim=1)
        g2 = F.relu(self.Cheby2(g2, self.A1))
        
        att = torch.cat([self.b*g1, self.c*g2], dim=1)
        out = torch.cat([g1, g2], dim=1)
        out = F.softmax(att, dim=1) * out

        out = out.reshape(x.shape[0], -1)
        out = self.HC(out)
        out = F.softmax(out, dim=1)
        
        return out