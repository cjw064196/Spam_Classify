from speech_fliter import createVocabList,setofWord2Vec
from TrainNB import trainNB
from testNB import classifyNB
from spam_classify import textParse
import re
import numpy as np
import random

def bagofWord2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList) #创建一个其中所含元素都为0的向量\
    for word in inputSet: #遍历每个词条
        if word in vocabList: #如果词条存在于词汇表中，则计数加1
            returnVec[vocabList.index(word)]+=1
    return returnVec #返回词袋模型

def spamtest():
    docList = []
    classList = []
    fullText=[]
    for i in range(1, 26):
        wordList = textParse(
            open('/home/cjw/PycharmProjects/Spam_classification/email/spam/%d.txt' % i, encoding='ISO-8859-1').read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(
            open('/home/cjw/PycharmProjects/Spam_classification/email/ham/%d.txt' % i, encoding='ISO-8859-1').read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#创建词汇表，不重复
    trainingSet=list(range(50)) #创建存储训练集的索引值的列表和测试集的索引值的列表
    testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]#创建训练集矩阵和训练集类别标签系向量
    trainClasses=[]
    for docIndex in trainingSet:#遍历训练集
        trainMat.append(setofWord2Vec(vocabList,docList[docIndex])) #将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])#将类别添加到训练集类别标签系向量中
    p0V,p1V,pSpam=trainNB(np.array(trainMat),np.array(trainClasses))
    errCount=0
    for docIndex in testSet:#遍历测试集
        wordVector=setofWord2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errCount+=1
            print('分类错误的测试集',docList[docIndex])
    print('错误率：%.2f%%'%(float(errCount)/len(testSet)*100))

if __name__=='__main__':
    spamtest()