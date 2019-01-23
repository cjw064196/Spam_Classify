from TrainNB import *
from functools import reduce

"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
    vec2Classify - 待分类的词条数组
    p0Vec - 非侮辱类的条件概率数组
    p1Vec -侮辱类的条件概率数组
    pClass1 - 文档属于侮辱类的概率
Returns:
    0 - 属于非侮辱类
    1 - 属于侮辱类
"""
def classifyNB(vec2classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2classify*p0Vec)+np.log(1-pClass1)
    print('p0:',p0)
    print('p1:',p1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    list0Posts,listClasses=loadDataSet()
    myVocabList=createVocabList(list0Posts)
    trainMat=[]
    for postinDoc in list0Posts:
        trainMat.append(setofWord2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB(np.array(trainMat),np.array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=np.array(setofWord2Vec(myVocabList,testEntry))
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setofWord2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')
    else:
        print(testEntry, '属于非侮辱类')

if __name__=='__main__':
    testingNB()
