"""
 * @Author: 汤达荣 
 * @Date: 2018-07-19 20:58:19 
 * @Last Modified by:   汤达荣 
 * @Last Modified time: 2018-07-19 20:58:19 
 * @Email: tdr1991@outlook.com 
""" 
#coding:utf-8


import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName, "r")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataSet, labels):
    dataMat = np.mat(dataSet)
    labelMat = np.mat(labels).transpose()
    m, n = np.shape(dataMat)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = (labelMat - h) # 这里使用的是交叉熵
        # 如果使用平方误差，则是以下公式，其中的相乘是逐元素相乘
        #error = (labelMat - h) * h * (1 - h)
        weights = weights + alpha * dataMat.transpose() * error
    return weights

def stocGradAscent(dataMat, labelMat):
    m, n = np.shape(dataMat)
    alpha = 0.01
    weight = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMat[i] * weight))
        error = labelMat[i] - h
        weight = weight + alpha * error * np.array(dataMat[i])
    return weight

def stocGradAscentModify(dataMat, labelMat, numIter=150):
    m, n = np.shape(dataMat)
    weight = np.ones(n)
    for i in range(numIter):
        dataIndex = list(range(m))
        for j in range(m):
            alpha = 4 / (1.0 + i + j) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMat[randIndex] * weight))
            error = labelMat[randIndex] - h
            weight = weight + alpha * error * np.array(dataMat[randIndex])
            del(dataIndex[randIndex])
    return weight

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1
    else:
        return 0

def colicTest():
    frTrain = open("../data/horseColicTraining.txt", "r")
    frTest = open("../data/horseColicTest.txt", "r")
    trainingSet = []
    trainingLabel = []
    for line in frTrain.readlines():
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabel.append(float(currLine[-1]))
    trainWeights = stocGradAscentModify(np.array(trainingSet), trainingLabel, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: {}".format(errorRate))
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after {} iterations the average error rate is {}".format(numTests, errorSum / float(numTests)))
    
def plotBestFit(wei):
    #weights = wei.getA()
    dataMat, labelMat = loadDataSet("../data/testSet.txt")
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c="red", marker="s")
    ax.scatter(xcord2, ycord2, s=30, c="yellow")
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

if __name__ == "__main__":
    #dataMat, labelMat = loadDataSet("../data/testSet.txt")
    #weights = gradAscent(dataMat, labelMat)
    #weights = stocGradAscent(dataMat, labelMat)
    #weights = stocGradAscentModify(dataMat, labelMat)
    #plotBestFit(weights)
    multiTest()