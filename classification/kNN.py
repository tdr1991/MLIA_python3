"""
 * @Author: 汤达荣 
 * @Date: 2018-07-18 10:12:05 
 * @Last Modified by:   汤达荣 
 * @Last Modified time: 2018-07-18 10:12:05 
 * @Email: tdr1991@outlook.com 
""" 
#coding:utf-8
import operator

import numpy as np
import matplotlib.pyplot as plt

# k-Nearst Neighbors algorithm
def kNN(inX, dataSet, labels, k):
    """
    inX：需要预测的标签的数据
    dataSet：对比数据集
    labels：对比数据集标签
    k：k近邻的k
    """
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet # tile将inX复制dataSetSize次，再与数据集相减，此处可以对比for循环
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(fileName):
    fr = open(fileName, "r")
    cont = fr.readlines()
    numberOfLines = len(cont)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in cont:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector

def classTest():
    hoRatio = 0.1
    dataSet, labels = file2matrix("../data/datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(dataSet)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierRes = kNN(normMat[i, :], normMat[numTestVecs:m, :], labels[numTestVecs:m], 3)
        print("预测类标：%s, 真实类标：%s" % (classifierRes, labels[i]))
        if classifierRes != labels[i]:
            errorCount += 1.0
    print("错误率为：%f" % (errorCount / float(numTestVecs)))


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def draw(x, y, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dupuplicationLabel = list(set(labels))
    print(dupuplicationLabel)
    labels = [dupuplicationLabel.index(l) for l in labels]
    labels = np.array(labels)
    print(labels[:5])
    ax.scatter(x, y, 15.0 * labels, 15.0 * labels)
    plt.show()


if __name__ == "__main__":
    """
    dataSet, labels = file2matrix("../data/datingTestSet.txt")
    
    print(dataSet[:3, :])
    print(labels[:3])
    draw(dataSet[:, 1], dataSet[:, 2], labels)
    print(labels[:3])
    
    normMat, ranges, minVals = autoNorm(dataSet)
    print(normMat[:3, :])
    """

    #classTest()
    a = {"a":1, "b":3, "c":0}
    #b = sorted(a.items(), key=operator.itemgetter(1), reverse=True)
    b = sorted(a.items(), key=operator.itemgetter(1), reverse=True)
    print(a.items())
    print(b)

    