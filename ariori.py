# -*- coding: utf-8 -*-
"""
Refer to:https://www.cnblogs.com/lsqin/p/9342926.html 
"""
import seaborn as sns
import pandas as pd
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
def loadDataSet():
    return [["面包","可乐","麦片"], ["牛奶", "可乐"], ["牛奶", "面包", "麦片"],
            ["牛奶", "可乐"],["面包","鸡蛋","麦片"],["牛奶","面包","可乐"],
            ["牛奶","面包","鸡蛋","麦片"],["牛奶","面包","可乐"],["面包","可乐"]]

def createC1(dataSet):
    C1 = []   
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                C1.sort()
    return list(map(frozenset, C1)) 
def scanD(dataSet, Ck, minSupport):
    ssCnt = {}   
    for tid in dataSet:
        for can in Ck:
            if can.issubset(tid):
                ssCnt[can] = ssCnt.get(can, 0) + 1  
    numItems = float(len(dataSet))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)  
        supportData[key] = support
    #print(retList, supportData)
    return retList, supportData  
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]   
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)  
    D = list(map(set, dataSet))  
    L1, supportData = scanD(D, C1, minSupport)  
    L = [L1]  
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)  
        Lk, supK = scanD(D, Ck, minSupport)  
        L.append(Lk);supportData.update(supK) 
        k += 1
    return L, supportData

def draw(L,supportData):
    data=[]
    x = ['面包','可乐','麦片','鸡蛋','牛奶']
    for i in range(5):
        for j in range(5):
            y=[]
            y.append(x[i])
            y.append(x[j])
            k = {x[i],x[j]}
            if x[i]==x[j]:
                y.append(1)
            elif k in L:
                for l in range(len(L)):
                    if L[l] == k:
                        y.append(supportData[L[l]])           
            else:
                y.append(0)
            data.append(y)
    df=pd.DataFrame(data,columns=['y','x','value'])
    df=df.pivot(index='y',columns='x',values='value')
    return sns.heatmap(df,linewidths=0.5)

def CAL(L,supportData,minConfidence):
    n = len(L)
    for i in range(1,n):
        for j in L[i]:
            m = len(j)
            for k in range(m-1):
                for d in L[k]:
                    if d.issubset(j):
                        conf= supportData[j]/supportData[d]
                        lift= conf/supportData[j-d]
                        if conf > minConfidence:
                            x=str(list(d)).replace('\'','').replace('[','').replace(']','')
                            y=str(list(j-d)).replace('\'','').replace('[','').replace(']','')
                            z=str(list(j)).replace('\'','').replace('[','').replace(']','')
                            print(x,'——>',y,'support(%s)=%f'%(z,supportData[j]),'support(%s)=%f'%(x,supportData[d]),'support(%s)=%f'%(y,supportData[j-d]),'conf=%f'%conf,'lift=%f'%lift)
if __name__=='__main__':
    dataSet = loadDataSet()
    L, supportData = apriori(dataSet,minSupport=0.2)
    draw(L[1],supportData)
    CAL(L,supportData,minConfidence=0.5)
