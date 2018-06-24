# -*- coding: utf-8

import numpy as np
from collections import Counter
import operator
import matplotlib
import matplotlib.pyplot as plt
def file2matrix(filename):
    """
    desc:
    导入训练数据
    :param filename: filename：数据文件路径
    :return: returnMat:数据矩阵 classLabelVector：对应类别
    """
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)  # 为什么又要调用一次? 文件读取再好好看一看
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


datingDataMat, datingLabels = file2matrix('dataset.txt')
# print(datingDataMat)
# print(datingLabels)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
# plt.show()
def autoNorm(dataSet):
    """
    desc:归一化特征值，消除特征之间量级不同导致的影响
    :param dataSet:
    dataset：数据集
    :return:
    归一化后的数据集normDataSet, ranges和minVals即最小值与范围
    归一化公式：
    Y = (X - min)/(Xmax - Xmin)
    """
    minVals = dataSet.min(0)
    # print(minVals)   # 每一列的最小值，一共三列
    maxVals = dataSet.max(0)
    # print(maxVals)  # 每一列的最大值，一共三列
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]  # 1000行
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # np.tile()重复minval这一行1000次，（m,1）中的1代表每一行只重复1次 X-min
    # print(np.tile(minVals, (m, 2)))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))   # np.tile(ranges, (m, 1))重复1000次
    return normDataSet, ranges, minVals


normMat, ranges, minVals = autoNorm(datingDataMat)
# print(normMat)

def datingClassTest():
    """
    desc:对约会网站的测试方法
    :return:
    """
    hoRatio = 0.1
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, 0], normMat[numTestVecs:m, :], normMat[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1
    print("the total error rate is: %f" % (errorCount /float(numTestVecs)))
    print(errorCount)


def classify0(inX, dataSet, labels, k):
    """
    inx[1,2,3]
    DS=[[1,2,3],[1,2,0]]
    inX: 用于分类的输入向量
    dataSet: 输入的训练样本集
    labels: 标签向量
    k: 选择最近邻居的数目
    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.
    预测数据所在分类可在输入下列命令
    kNN.classify0([0,0], group, labels, 3)
    """
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def clasdiperson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("you will probably like this person:", resultList[classifierResult-1])


clasdiperson()

