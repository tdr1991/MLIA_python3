"""
 * @Author: 汤达荣 
 * @Date: 2018-07-19 11:25:48 
 * @Last Modified by:   汤达荣 
 * @Last Modified time: 2018-07-19 11:25:48 
 * @Email: tdr1991@outlook.com 
""" 
#coding:utf-8

import re
import random

import numpy as np

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % (word))
    return returnVec

def trainBayes(trainMat, trainCategory):
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])
    #pAbusive = sum(trainCategory) / float(numTrainDocs)
    unique = list(set(trainCategory))
    categoryCount = dict()
    for u in unique:
        categoryCount[u] = 0
    for c in trainCategory:
        categoryCount[c] += 1
    pCategory = dict()
    for key in categoryCount.keys():
        pCategory[key] = categoryCount[key] / float(numTrainDocs)
    pNum = np.ones((len(unique),numWords))
    #p1Num = np.ones(numWords)
    #p0Demon = 2.0
    #p1Demon = 2.0
    pDemon = np.tile([2.0], len(unique))
    pDemon = np.reshape(pDemon, (len(unique), 1))
    for i in range(numTrainDocs):
        index = unique.index(trainCategory[i])
        pNum[index] += trainMat[i]
        pDemon[index, 0] += 1.0
        """
        if trainCategory[i] == 1:
            p1Num += trainMat[i]
            p1Demon += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Demon += sum(trainMat[i])
        """
    #p1Vect = np.log(p1Num / p1Demon)
    #p0Vect = np.log(p0Num / p0Demon)
    pVect = np.log(pNum / pDemon)   #以e为底的对数
    #return p0Vect, p1Vect, pAbusive
    return pVect, unique, pCategory
"""
def classifyBayes(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
"""

def classifyBayes(vec2Classify, pVect, unique, pCategory):
    vec2Classify = np.reshape(vec2Classify, (len(vec2Classify), 1))
    mt = np.dot(pVect, vec2Classify)
    mt = np.reshape(mt, mt.shape[0])
    p = mt + np.log([pCategory[u] for u in unique])
    index = np.unravel_index(np.argmax(p), p.shape)[0]
    return unique[index]
    """
    #p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    """
def textParse(bigString):
    listOfTokens = re.split(r"\W*", bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open("./data/email/spam/%d.txt" % (i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open("./data/email/ham/%d.txt" % (i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testingSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testingSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    pVect, unique, pCategory = trainBayes(np.array(trainMat), np.array(trainClasses))
    errorCount = 0.0
    for docIndex in testingSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyBayes(np.array(wordVector), pVect, unique, pCategory) != classList[docIndex]:
            errorCount += 1
    print("the error rate is: {}".format(float(errorCount) / len(testingSet)))

if __name__ == "__main__":
    spamTest()
 