"""
 * @Author: 汤达荣 
 * @Date: 2018-07-18 21:21:27 
 * @Last Modified by:   汤达荣 
 * @Last Modified time: 2018-07-18 21:21:27 
 * @Email: tdr1991@outlook.com 
""" 
#coding:utf-8

import math
import operator

import matplotlib.pyplot as plt
import numpy as np

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCoutns = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCoutns.keys():
            labelCoutns[currentLabel] = 0
        labelCoutns[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCoutns.keys():
        prob = float(labelCoutns[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息熵越大代表越不确定，所以找最佳划分特征时需要找能使不确定性降到最低的
        infoGain = baseEntropy - newEntropy # 与标准熵相差最大的
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = dict()
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeature(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    featValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def classifierTest():
    fr = open("../data/lenses.txt", "r")
    dataSet = [example.strip().split("\t") for example in fr.readlines()]
    labels = ["age", "prescript", "astigmatic", "tearRate"]
    featLabels = [feat for feat in labels]
    #dataSet = np.array(dataSet)
    trainData = dataSet[:-1]
    testData = dataSet[-1]
    lensesTree = createTree(dataSet, labels)
    print(lensesTree)
    print(featLabels)
    print(labels)
    pred = classify(lensesTree, featLabels, testData[:-1])
    print(pred)


if __name__ == "__main__":
    """
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    print(calcShannonEnt(dataSet))
    print(splitDataSet(dataSet, 0, 0))
    print(splitDataSet(dataSet, 0, 1))
    
    a = [1,2,3]
    print(a[:-1])
    
    classifierTest()
    """
    a = [[1,2,3], [3, 4, 5]]
    b = [[2.0], [2.0]]
    #b = [2.0, 2.0]
    c = [[1,2,3], [3, 4, 5]]
    a = np.array(a)
    b = np.array(b)
    print(a / b)
    print(a * b)
    print(a * c)
    
    print(np.log(2))
    