# -*- coding: UTF-8 -*-
import re
def textParse(bigString):
    listofTokens=re.split(r'\W*',bigString)
    return [tok.lower() for tok in listofTokens if len(tok)>2]

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

if __name__=='__main__':
    docList=[]
    classList=[]
    for i in range(1,26):
        wordList=textParse(open('/home/cjw/PycharmProjects/Spam_classification/email/spam/%d.txt' % i,encoding='ISO-8859-1').read())
        docList.append(wordList)
        classList.append(1)
        wordList=textParse(open('/home/cjw/PycharmProjects/Spam_classification/email/ham/%d.txt' % i,encoding='ISO-8859-1').read())
        docList.append(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    print(vocabList)
