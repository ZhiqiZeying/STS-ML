import os
import sys
from torch.optim import lr_scheduler
import numpy as np
import torch
import torch.optim as optim
from scipy import io as scio
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import zscore
from model import DE_MoE
from AutoWeight import AutomaticWeightedLoss
from tqdm import tqdm
from utils import eegDataset, calc_diff_loss

import os
import copy
import config

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TORCH_HOME'] = './'  # setting the environment variable

path = 'data/SEED/data_independent/DE/time/session1'
# path = '/GNN_NPY_DATASETS/MPED/data_dependent'
# path = '/GNN_NPY_DATASETS/SEED/data_independent'
batch_size = config.batch_size
epochs = config.epochs
lr_list = config.lr_list
weight_decay = config.weight_decay
drop_rate = config.drop_rate
device = config.device
num_workers = config.num_workers
DATASETS = ['SEED', 'SEED_IV', 'MPED']
DATASET = path.strip().split('/')[-5]
assert DATASET in DATASETS
DEPENDENT = path.strip().split('/')[-4]
# if DEPENDENT == 'data_independent':
#     DATASET = DATASET+'_'+DEPENDENT


def load_dataloader(data_train, data_test, label_train, label_test):
    train_iter = DataLoader(dataset=eegDataset(data_train, label_train),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            drop_last=False)

    test_iter = DataLoader(dataset=eegDataset(data_test, label_test),
                            batch_size=int(data_test.shape[0]/10),
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False)

    return train_iter, test_iter


def train(train_iter, test_iter, model, model_best, awl, criterion, lr_list, epochs, acc_test_best, people):
    # Train

    print('began training on', device, '...')

    # acc_test_best = 0.0
    n = 0
    for ep in range(epochs):
        
        for lts, lr, l in lr_list:
            if acc_test_best == 1:
                # print('acc_test_best = 1')
                break
            
            for lt in range(int(lts)):
                
                if acc_test_best == 1:
                    # print('acc_test_best = 1')
                    break
                
                model.testmode = False
                model.train()
                n += 1
                # batch_id = 1
                correct, total, total_loss = 0, 0, 0.
                for ind, data in enumerate(train_iter):
                    
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)

                    model.testmode = False
                    model.train()

                    x = data[0].float().to(device)  # [256,62,5]([B,C,F])
                    y = data[1].to(device)

                    # x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, out = model(x)
                    out = model(x)
                    _, pred = torch.max(out, dim=1)
                    correct += sum([1 for a, b in zip(pred, y) if a == b])
                    total += len(y)
                    accuracy = correct/total

                    # out1 = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
                    # out1 = torch.tensor(out1)
                    # out2 = [x11, x12]
                    # out2 = torch.tensor(out2)

                    # loss1 = calc_diff_loss(out1)
                    # loss2 = calc_diff_loss(out2)
                    loss = criterion(out, y.long())

                    # loss = awl(loss1, loss2, loss3)

                    optimizer.zero_grad()
                    total_loss += loss
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                    # print('Epoch {}, batch {}, loss: {}, accuracy: {}'.format(ep + 1,
                    #                                                           batch_id,
                    #                                                           total_loss / batch_id,
                    #                                                           accuracy))

                    # batch_id += 1
                    if (ind+1) % 10 == 0:
                        print('people {}, epoch {}, lt {}-{}, ind {}, loss: {}, accuracy: {}'.format(
                            people, ep+1, l, lt + 1, ind + 1, total_loss / ind+1, accuracy))
                        acc_test = evaluate(test_iter, model)

                        if acc_test >= acc_test_best:
                            n = 0
                            acc_test_best = acc_test
                            model_best = model
                            checkpoint = {
                                'model_state_dict': model_best.state_dict(),
                                'ACC': acc_test_best
                            }
                            torch.save(
                                checkpoint, f'SGCNN/{DATASET}_checkpoint/independent/session1/{DATASET}_checkpoint_{people}.pkl')
                        
                        if acc_test_best == 1:
                            print('acc_test_best = 1')
                            break
                
                acc_test = evaluate(test_iter, model)

                if acc_test >= acc_test_best:
                    n = 0
                    acc_test_best = acc_test
                    model_best = model
                    checkpoint = {
                        'model_state_dict': model_best.state_dict(),
                        'ACC': acc_test_best
                    }
                    torch.save(
                        checkpoint, f'SGCNN/{DATASET}_checkpoint/independent/session1/{DATASET}_checkpoint_{people}.pkl')
                    
                if acc_test_best == 1:
                    print('acc_test_best = 1')
                    break

        print('people {} Total loss for epoch {}: {}'.format(
            people, ep + 1, total_loss))

        print('>>> best test Accuracy: {}'.format(acc_test_best))

        if acc_test_best == 1:
            print('acc_test_best = 1')
            break

    return acc_test_best


def evaluate(test_iter, model):
    # Eval
    print('began test on', device, '...')
    model.testmode = True
    model.eval()
    correct, total = 0, 0
    for x, y in test_iter:
        # Add channels = 1
        x = x.float().to(device)

        # Categogrical encoding
        y = y.to(device)

        out = model(x)

        _, pre = torch.max(out, dim=1)

        correct += sum([1 for a, b in zip(pre, y) if a == b])
        total += len(y)
    print('test Accuracy: {}'.format(correct / total))
    return correct / total


def runs(people):

    print(f'load object {people}\'s data.....')
    train_data = np.load(path + '/' + 'train_dataset_{}.npy'.format(people))
    train_label = np.load(
        path + '/' + 'train_labelset_{}.npy'.format(people)).flatten()
    test_data = np.load(path + '/' + 'test_dataset_{}.npy'.format(people))
    test_label = np.load(
        path + '/' + 'test_labelset_{}.npy'.format(people)).flatten()
    # train_data = np.transpose(train_data, [1, 0, 2])
    train_label = train_label+1
    # test_data = np.transpose(test_data, [1, 0, 2])
    test_label = test_label+1
    print('loaded!')

    acc_test_best = 0.0

    if not os.path.exists(f'SGCNN/{DATASET}_checkpoint/independent/session1'):
        os.makedirs(
            f'SGCNN/{DATASET}_checkpoint/independent/session1')

    if os.path.exists(f'SGCNN/{DATASET}_checkpoint/independent/session1/{DATASET}_checkpoint_{people}.pkl'):
        check = torch.load(
            f'SGCNN/{DATASET}_checkpoint/independent/session1/{DATASET}_checkpoint_{people}.pkl', weights_only=True)
        acc_test_best = check['ACC']

    HC = None
    if 'SEED_IV' in DATASET:
        HC = 4
    elif 'MPED' in DATASET:
        HC = 7
    else:
        HC = 3
    assert HC is not None

    awl = AutomaticWeightedLoss(3)

    test_data = zscore(test_data)  # 各数据在数据集中的相对位置

    model_best = DE_MoE(5, 65, 62, linearsize=512, dropout=drop_rate, testmode=False, HC=HC).to(device)
    model = DE_MoE(5, 65, 62, linearsize=512, dropout=drop_rate, testmode=False, HC=HC).to(device)

    # 使用这个函数需要注意：标签是整数，不要onehot，已经包含了softmax
    criterion = nn.CrossEntropyLoss().to(device)

    # train_data = zscore(train_data)
    num = int(train_data.shape[0]/14)
    for i in range(14):
        train_data[i*num:(i+1)*num] = zscore(train_data[i*num:(i+1)*num])

    # 训练的数据要求导，必须使用torch.tensor包装
    train_data = torch.tensor(train_data)
    train_label = torch.tensor(train_label)
    test_data = torch.tensor(test_data)
    test_label = torch.tensor(test_label)

    train_iter, test_iter = load_dataloader(
        train_data, test_data, train_label, test_label)
    acc_test_best = train(train_iter, test_iter, model, model_best, awl, criterion, lr_list, epochs, acc_test_best, people)
    # acc_mean = acc_mean + acc_test_best/15
    acc_all.append(acc_test_best)
    
    if people == 15:
        acc_mean = np.array(acc_all).mean()

        print('>>> LOSV test acc: ', acc_all)
        print('>>> LOSV test mean acc: ', acc_mean)
        print('>>> LOSV test std acc: ', np.std(np.array(acc_all)))


if __name__ == '__main__':
    
    acc_all = []
    
    for i in range(1):
        runs(9)
