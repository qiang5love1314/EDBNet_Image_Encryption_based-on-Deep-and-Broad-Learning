from BLS_Regression import *
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy import ndimage
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchstat import stat
from hashlib import sha3_256
import time

np.random.seed(0)
torch.manual_seed(8)
batchsize = 10
imagePath = './Lena.png'
a = 10
b = 28
c = 8/3

# 读取图片
def getInputImage(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, -1)
    w,h = img.shape[:2]
    return img,w,h

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

# 定义数据集类
class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        self.root_dir = root_dir
        self.txt_path = [txt_COVID,txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.img_list[idx][1])}
        return sample

# 生成数据集
def generateDataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = transforms.Compose([ 
    transforms.Resize(256),
    transforms.RandomResizedCrop((256),scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])
    val_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize])

    trainset = CovidCTDataset(root_dir='/Users/zhuxiaoqiang/Desktop/image encryption',
                              txt_COVID='CT_COVID/trainCT_COVID.txt',
                              txt_NonCOVID='CT_NonCOVID/trainCT_NonCOVID.txt',
                              transform= train_transformer)
    valset =  CovidCTDataset( root_dir='/Users/zhuxiaoqiang/Desktop/image encryption',
                              txt_COVID='CT_COVID/valCT_COVID.txt',
                              txt_NonCOVID='CT_NonCOVID/valCT_NonCOVID.txt',
                              transform= val_transformer)
    testset = CovidCTDataset( root_dir='/Users/zhuxiaoqiang/Desktop/image encryption',
                              txt_COVID='CT_COVID/testCT_COVID.txt',
                              txt_NonCOVID='CT_NonCOVID/testCT_NonCOVID.txt',
                              transform= val_transformer)

    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=True, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=True, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=True, shuffle=True)
    
    return train_loader, val_loader, test_loader

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # 卷积层，检测横向特征
#         self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
#         # 卷积层，检测纵向特征
#         self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
#         # 卷积层，检测边缘特征
#         self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
#         self.dropout = nn.Dropout(0.5)
#         # 池化层，减小图像尺寸
#         self.pool = nn.MaxPool2d(2, 2)
#         # 全连接层，输出二分类结果
#         self.fc = nn.Linear(128 * 32 * 32, 2)
  
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 128 * 32 * 32)
#         x = self.fc(x)
#         return x

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # First convolutional layer with sobel for horizontal
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1.weight.data = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]])
        self.conv1.bias.data.fill_(0)
        self.conv1 = nn.ReLU(self.conv1)
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional layer with sobel for vertical
        self.conv2 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2.weight.data = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]])
        self.conv2.bias.data.fill_(0)
        self.conv2 = nn.ReLU(self.conv2) 
        
        self.fc1 = nn.Linear(49152, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x1 = (self.conv1(x))
        x2 = self.pool(self.conv2(x1))
        x = x2
        x = x.view(x.size(0), -1)
        x = (self.fc1(x))
        x = (self.fc2(x))
        return x

# 训练模型
def trainModel():
    train_loader, val_loader, test_loader = generateDataset()
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(100):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data['img'], data['label']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch {} loss: {:.4f}'.format(epoch + 1, running_loss / len(train_loader)))

    print('Finished Training')
    torch.save(model.state_dict(), 'ConvNetnew2.pt')
    return model

def generateFeatureMatrix(weights, w, h):
    zoom_factor = tuple(np.array((w, h))) / np.array(weights.shape)
    fc_feature = ndimage.zoom(weights, zoom_factor, order=2, mode='nearest')
    return fc_feature

def hashValue(data):
    hash_object = sha3_256(data)
    return hash_object

def getFractionalData(x):
    decimal_point = x - int(x)
    return decimal_point

def lorenz(x, y, z, a=10, b=28, c=8/3):
    dx = -a * x + a * y
    dy = -x * z + b * x - y
    dz = x * y - c * z
    return dx, dy, dz

def generate_chaos_sequence(initial_values, iterations):
    x, y, z = initial_values
    chaos_sequence = []
    for i in range(iterations):
        dx, dy, dz = lorenz(x, y, z)
        x, y, z = x + dx * 0.01, y + dy * 0.01, z + dz * 0.01
        chaos_sequence.append(x)
    return chaos_sequence[:iterations]

def getDataAndLabel(loader):
    datalist = []
    labelList = []
    for batch_index, batch_samples in enumerate(loader):      
        dataTarget, target = batch_samples['img'], batch_samples['label']
        # dataTarget = dataTarget.view(dataTarget.size(0), -1)
        # target = target.view(target.size(0), -1)
        datalist.append(dataTarget.numpy().flatten())
        labelList.append(target.numpy().flatten())
    datalist = np.concatenate(datalist, axis=0)
    labelList = np.concatenate(labelList, axis=0)
    return datalist, labelList

def updateFeatures(feature):
    feature = 3.6789 * feature * (1 - feature)
    return np.round(np.abs(feature) * 10 **6)

# 主程序
if __name__ == '__main__':
    # img, w, h = getInputImage(imagePath)
    # plt.imshow(img)
    # plt.show()
    start_time = time.time()
    w = 256
    h = 256

    # 加载训练集、测试集、验证集
    train_loader, val_loader, test_loader = generateDataset()
    # myModel = trainModel()
    
    # for batch_index, batch_samples in enumerate(train_loader):      
    #     dataTarget, target = batch_samples['img'], batch_samples['label']

    # 加载已训练模型
    myModel = ConvNet()
    myModel.load_state_dict(torch.load('ConvNet.pt'))
    # stat(myModel,(3, 256, 256))
    # 切换到评估模式
    myModel.eval()
    end_time1 = time.time()
    # 迭代测试数据
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            testInputs, labels = data['img'], data['label']
            outputs = myModel(testInputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算准确率
    accuracy = 100 * correct / total
    # print('Accuracy of the model on the test set: {:.2f}%'.format(accuracy))

    # 评估模型
    test_loss = 0
    correct = 0
    results = []
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    criteria = nn.CrossEntropyLoss()
    # Don't update model
    # with torch.no_grad():
    #     tpr_list = []
    #     fpr_list = []
        
    #     predlist=[]
    #     scorelist=[]
    #     targetlist=[]
    #     # Predict
    #     for batch_index, batch_samples in enumerate(test_loader):
    #         data, target = batch_samples['img'], batch_samples['label']
    #         output = myModel(data)
    #         test_loss += criteria(output, target.long())
    #         score = F.softmax(output, dim=1)
    #         pred = output.argmax(dim=1, keepdim=True)
    #         correct += pred.eq(target.long().view_as(pred)).sum().item()
    #         targetcpu=target.long().cpu().numpy()
    #         predlist=np.append(predlist, pred.cpu().numpy())
    #         scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
    #         targetlist=np.append(targetlist,targetcpu)
        
    #     TP = ((predlist == 1) & (targetlist == 1)).sum()
    #     TN = ((predlist == 0) & (targetlist == 0)).sum()
    #     FN = ((predlist == 0) & (targetlist == 1)).sum()
    #     FP = ((predlist == 1) & (targetlist == 0)).sum()

        # print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        # print('TP+FP',TP+FP)
        # p = TP / (TP + FP)
        # print('precision',p)
        # p = TP / (TP + FP)
        # r = TP / (TP + FN)
        # print('recall',r)
        # F1 = 2 * r * p / (r + p)
        # acc = (TP + TN) / (TP + TN + FP + FN)
        # print('F1',F1)
        # print('acc',acc)
    start_time2 = time.time()
    randomNumber = np.random.randint(0, 255)
    targetImage, width, heigh = getInputImage('results/5.png')
    tragetLabel = 0
    targetImage = cv2.resize(targetImage, [w, h])
    # cv2.imwrite('results/plain1.png', (targetImage))
    horizontal, vertical = (100+randomNumber) % w, (100+randomNumber) % h
    plaintextPixel = targetImage[horizontal, vertical]      # let chaotic value be different by input image
    
    initialKey = hashValue('898oaFs09f'.encode('utf8'))     # initial hash value which I set it
    initialKey.update(str(plaintextPixel).encode('utf8'))   # I put a pixel value here
    key_hash = initialKey.hexdigest()
    print('initialKey', key_hash)
    # generate two random chaos sequences using hash function and plaintext pixel
    initialNumber1 = getFractionalData(int(key_hash[1:5], 16) / (w * h))    # generate initial number from hash value for chaotic sequence
    initialNumber2 = getFractionalData(int(key_hash[6:10], 16) / (w * h))
    initialNumber3 = getFractionalData(int(key_hash[11:15], 16) / (w * h))
    chaotic_seq1 = generate_chaos_sequence([initialNumber1, initialNumber2, initialNumber3], w)
    chaotic_seq1 = np.round(np.abs(chaotic_seq1) * 10 **6) % w
    chaotic_seq2 = []
    initialLogistic = getFractionalData(chaotic_seq1[-1] / (w * h))
    for i in range(h):
        initialLogistic = 3.6789 * initialLogistic * (1 - initialLogistic)
        chaotic_seq2.append(initialLogistic)
    chaotic_seq2 = np.round(np.abs(chaotic_seq2) * 10 **6) % h

    # scramble the image
    row_permutation = np.arange(w)
    col_permutation = np.arange(h)
    index1 = np.argsort(chaotic_seq1)
    row_permutation = row_permutation[index1]
    index2 = np.argsort(chaotic_seq2)
    col_permutation = col_permutation[index2]
    scrambled_img = targetImage[row_permutation, :]
    scrambled_img = scrambled_img[:, col_permutation]
    
    for i in range(w):
        scrambled_img[i, :] = np.roll(scrambled_img[i, :], index1[i])       # cyclic right shift
        for j in range(h):
            scrambled_img[:, j] = np.roll(scrambled_img[:, j], -index2[j])  # cyclic left shift
    # cv2.imshow('scramble',scrambled_img)
    # cv2.imwrite('scrambled_img.png', (scrambled_img))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    end_time2 = time.time()
    # BLS training procedure
    trainData, trainLabel = getDataAndLabel(train_loader)
    testData, testLabel = getDataAndLabel(test_loader)
    trainData = np.reshape(abs(trainData),(len(trainLabel), 3*w*h))
    testData = np.reshape(abs(testData),(len(testLabel), 3*w*h))
    # model parameters
    NumFea = 20
    NumWin = 3
    NumEnhan = 110
    s = 0.9  # shrink coefficient
    C = 2 ** -10  # Regularization coefficient
    np.seterr(invalid='ignore')
    NetoutTrain, NetoutTest, WeightTop, T3 = bls_regression(trainData, trainLabel, testData, testLabel, s, C, NumFea, NumWin, NumEnhan)
    start_time3 = time.time()
    NetoutTest = (NetoutTest > 0.5).astype(int).flatten()
    # np.savetxt('T3.txt', T3)
    # calculate the precision function
    equal_array = testLabel == NetoutTest
    num_equal = np.count_nonzero(equal_array)
    accuracy = num_equal / len(testLabel)
    
    BLSFeaturesNetwork = np.round(abs(generateFeatureMatrix(T3, w, h)) * 10 **5) % 256
    print(accuracy) # 20 3 110 0.9  2 ** -10   0.685

    # get the F1 and F2 parameters of CNN
    fc1_weight = None
    fc2_weight = None
    for name, params in myModel.named_parameters():
        if "fc1" in name and "weight" in name:
            fc1_weight = params.data.numpy()      # (32, 49152)
        elif "fc2" in name and "weight" in name:
            fc2_weight = params.data.numpy()      # (2, 32)
            break
    # generate the F1 and F2 features with the same size of plaintext image, and network matrix within [0, 255]
    fc1_feature = generateFeatureMatrix(fc1_weight, w, h)
    fc2_feature = generateFeatureMatrix(fc2_weight, w, h)
    Conv1FeaturesNetwork = updateFeatures(fc1_feature) % 256
    Conv2FeaturesNetwork = updateFeatures(fc2_feature) % 256
    # xor diffusion encryption
    if tragetLabel == 0:
        imgXOR = scrambled_img.astype('int') ^ BLSFeaturesNetwork.astype('int') ^ Conv1FeaturesNetwork.astype('int')
    else:
        imgXOR = scrambled_img.astype('int') ^ BLSFeaturesNetwork.astype('int') ^ Conv2FeaturesNetwork.astype('int')
    # cv2.imshow('imgXOR', np.uint8(imgXOR))
    # cv2.waitKey(0)
    # cv2.imwrite('results/salt-encryption.png', (imgXOR))
    end_time3 = time.time()
    run_time = end_time3 - start_time3 + end_time2 - start_time2 + end_time1 - start_time
    print("程序运行时间为：{:.4}".format(run_time), "秒")