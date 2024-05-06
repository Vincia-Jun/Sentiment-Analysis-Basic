"""
Author: Jun
Date  : 2024-04-26
"""
import jieba
import pandas as pd
from tqdm import tqdm
import sys
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, f1_score
from thop import profile

import sys
sys.path.insert(0,"models")

from TextCNN import TextCNN
from TextRCNN import TextRCNN
from TextRNN import TextRNN
from TextRNNAtt import TextRNNAtt
from TransformerText import TransformerText


# 构建词汇表
def get_vocab(train_data):
    vocab = set()
    cut_docs = train_data.text_a.apply(lambda x: jieba.cut(x)).values
    for doc in cut_docs:
        for word in doc:
            if word.strip():
                vocab.add(word.strip())

    # 将词表写入本地vocab.txt文件
    with open('data/vocab.txt', 'w') as file:
        for word in  vocab:
            file.write(word)
            file.write('\n')
    return vocab


# 定义配置函数
class Config():
    embedding_dim = 300 # 词向量维度
    max_seq_len = 200   # 文章最大词数 200
    vocab_file = 'data/vocab.txt' # 词汇表文件路径
    vocab_len = 35093


# 定义预训练类
class Preprocessor():
    def __init__(self, config):
        self.config = config
        # 初始化词和id的映射词典，预留0给padding字符，1给词表中未见过的词
        token2idx = {"[PAD]": 0, "[UNK]": 1} # {word：id}
        with open(config.vocab_file, 'r') as reader:
            for index, line in enumerate(reader):
                token = line.strip()
                token2idx[token] = index+2
                
        self.token2idx = token2idx
        
    def transform(self, text_list):
        # 文本分词，并将词转换成相应的id, 最后不同长度的文本padding长统一长度，后面补0
        idx_list = [[self.token2idx.get(word.strip(), self.token2idx['[UNK]']) for word in jieba.cut(text)] for text in text_list]
        tensor_list = [torch.tensor(sublist) for sublist in idx_list]
        padded_sequences = [F.pad(sequence, (0, self.config.max_seq_len - sequence.size(0))) for sequence in tensor_list]
        stacked_tensor = torch.stack(padded_sequences)
        
        return stacked_tensor
    


def train_one_epoch(model, optimizer, criterion, data_loader, epoch):
    
    model.train()
    optimizer.zero_grad()

    # for calculating loss and acc
    train_loss = torch.zeros(1).to(device)
    total_correct = 0
    total_samples = 0

     # for calculating accuracy
    data_loader = tqdm(data_loader, file = sys.stdout, ncols=100)
    data_loader.set_description(f"Epoch [{epoch}] [Train]")
    
    
    for step, data in enumerate(data_loader):

        x_train, y_train = data
        x_train, y_train = x_train.to(device), y_train.to(device)
        outputs = model(x_train)
        _, predicted = torch.max(outputs, 1)  # 获取预测的类别
        total_correct += (predicted == y_train).sum().item()  # 统计预测正确的数量
        total_samples += y_train.size(0)  # 统计样本总数

        loss = criterion(outputs, y_train)
        train_loss = (train_loss * step + loss.detach()) / (step + 1)  # update mean losses
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        data_loader.set_postfix(Loss=train_loss.item(), Acc=total_correct/total_samples)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    train_acc = total_correct / total_samples
    
    return train_loss, train_acc


def valid(model, criterion, data_loader, epoch):

    model.eval()
    eval_loss = torch.zeros(1).to(device)
    total_correct = 0
    total_samples = 0

     # for calculating accuracy
    data_loader = tqdm(data_loader, file = sys.stdout, ncols=100)
    data_loader.set_description(f"Epoch [{epoch}] [Valid]")

    with torch.no_grad():  
        for step, data in enumerate(data_loader):

            x_train, y_train = data
            x_train, y_train = x_train.to(device), y_train.to(device)
            outputs = model(x_train)
            _, predicted = torch.max(outputs, 1)  # 获取预测的类别

            total_correct += (predicted == y_train).sum().item()  # 统计预测正确的数量
            total_samples += y_train.size(0)  # 统计样本总数

            loss = criterion(outputs, y_train)
            eval_loss = (eval_loss * step + loss.detach()) / (step + 1)  # update mean losses
            data_loader.set_postfix(Loss=eval_loss.item(), Acc=total_correct/total_samples)

    eval_acc = total_correct / total_samples
    return eval_loss, eval_acc


def evaluate(model, x_test, y_test, epoch):
    model.eval()
    x_test, y_test = x_test.to(device), y_test.to(device)

    outputs = model(x_test)
    _, predicted = torch.max(outputs, 1)

    y_test, predicted = y_test.to('cpu'), predicted.to('cpu')
    test_acc = accuracy_score(y_test, predicted)
    test_f1  = f1_score(y_test, predicted)
    print(f"Epoch [{epoch}] [Test ]: test_acc: {test_acc}, test_f1: {test_f1}")
    return test_acc, test_f1


# TransformerText Config
class TTextConfig:
    def __init__(self):
        self.input_size = 35093
        self.d_model = 128
        self.n_heads = 2
        self.n_layers = 2
        self.dropout = 0.6
        self.max_length = 300
        self.d_ff = 64
        self.size = self.d_model
        self.num_classes = 2

if __name__ == '__main__':

    # parsers
    batch_size = 64
    num_workers = 4
    num_epochs = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get data
    train_data = pd.read_csv('data/train.tsv', sep='\t')
    valid_data = pd.read_csv('data/dev.tsv', sep='\t')
    test_data = pd.read_csv('data/test.tsv', sep='\t') 
    x_train, y_train = train_data.text_a.values, train_data.label.values # 训练集
    x_valid, y_valid = valid_data.text_a.values, valid_data.label.values # 验证集
    x_test, y_test = test_data.text_a.values, test_data.label.values # 测试集

    # get vocab
    vocab = get_vocab(train_data)
    config = Config()

    # dataloader, model, optimization
    preprocessor = Preprocessor(config)
    x_train = preprocessor.transform(x_train)
    y_train = torch.LongTensor(y_train)
    train_dataset = TensorDataset(x_train, y_train)

    x_valid = preprocessor.transform(x_valid)
    y_valid = torch.LongTensor(y_valid)
    valid_dataset = TensorDataset(x_valid, y_valid)

    x_test = preprocessor.transform(x_test)
    y_test = torch.LongTensor(y_test)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True,
                              pin_memory = True,
                              num_workers=num_workers)
    
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=batch_size, 
                              shuffle=True,
                              pin_memory = True,
                              num_workers=num_workers)
    
    test_loader = DataLoader(test_dataset, 
                              batch_size=batch_size, 
                              shuffle=True,
                              pin_memory = True,
                              num_workers=num_workers)
    
    # 1 build TextCNN
    # model = TextCNN(filter_sizes = [3,4,5],
    #                 num_classes = 2,
    #                 vocab_size = config.vocab_len,
    #                 num_filters = 128,
    #                 emb_dim = config.embedding_dim,
    #                 dropout = 0.4)


    # 2 build TextRCNN
    # model = TextRCNN(vocab_size = config.vocab_len,
    #                 emb_dim = config.embedding_dim,
    #                 hidden_size = 256,
    #                 num_classes = 2,
    #                 dropout = 0.4,
    #                 num_layers = 1)

    # 3 build TextRNN
    # model = TextRNN(vocab_size = config.vocab_len,
    #                 emb_dim = config.embedding_dim,
    #                 hidden_size = 128,
    #                 num_classes = 2,
    #                 num_layers = 2,
    #                 bidirectional = True,
    #                 dropout = 0.4)

    # 4 build TextRNNAtt
    # model = TextRNNAtt(vocab_size = config.vocab_len,
    #                     emb_dim = config.embedding_dim,
    #                     hidden_size1 = 256,
    #                     hidden_size2 = 64,
    #                     num_classes = 2,
    #                     num_layers = 2,
    #                     bidirectional = True,
    #                     dropout = 0.4)
    
    # 5 bulid Transformer
    ttconfig = TTextConfig()
    model = TransformerText(ttconfig)

    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    criterion = nn.CrossEntropyLoss()
    

    result_path = 'logs/%s_train_%d.csv' % (type(model).__name__, num_epochs)
    header = ['Epoch', 'Train_Loss', 'Test_Acc', 'Test_F1', 'Train_Acc', 'Eval_Loss', 'Eval_Acc']

    # 打开CSV文件并写入表头
    with open(result_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)


        # 初始化记录最大值的变量
        max_test_acc = 0
        max_test_f1 = 0
        max_eval_acc = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader, epoch)
            eval_loss, eval_acc = valid(model, criterion, valid_loader, epoch)
            test_acc, test_f1 = evaluate(model, x_test, y_test, epoch)
            # test_acc, test_f1  =  0 , 0

            # 更新最大值
            max_test_acc = max(max_test_acc, test_acc)
            max_test_f1 = max(max_test_f1, test_f1)
            max_eval_acc = max(max_eval_acc, eval_acc)

            data = [epoch, round(train_loss.item(), 2), round(test_acc*100, 2), round(test_f1*100, 2), 
                    round(train_acc*100, 2), round(eval_loss.item(), 2), round(eval_acc*100, 2)]
            writer.writerow(data)
            file.flush()
            print("----------------------------------------------------------------------------------------")


        # 找到最好的结果
        max_data = ['Max', '', round(max_test_acc*100, 2), round(max_test_f1*100, 2), 
                    '', '', round(max_eval_acc*100, 2)]
        writer.writerow(max_data)
        file.flush()


        # 计算模型时空复杂度
        inputs = torch.randint(config.vocab_len, (1, config.max_seq_len)).to(device)
        flops, params = profile(model, inputs=(inputs, ))

        print('Param = ' + str(params/1000**2) + 'M')
        print('FLOPs = ' + str(flops/1000**3) + 'G')
        
        modelPF = ['Param:', str(round(params/1000**2, 2)) + 'M',
                   'Flops:', str(round(flops/1000**3, 2)) + 'G']
        
        writer.writerow(modelPF)
