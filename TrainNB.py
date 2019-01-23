from speech_fliter import *
import numpy as np
def trainNB(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=np.ones(numWords)
    p1Num=np.ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=np.log(p1Num/p1Denom)
    p0Vect=np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    myVocabList=createVocabList(postingList)
    print('myVocabList:\n',myVocabList)
    trainMat=[]
    for postinDoc in postingList:
        trainMat.append(setofWord2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB(trainMat,classVec)
    print('p0V:\n',p0V)
    print('p1V:\n',p1V)
    print('pAb:\n',pAb)
    print('classVec:\n',classVec)