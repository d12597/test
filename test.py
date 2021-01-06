# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:39:36 2021

@author: 80686
"""


import pymysql
import time 
import numpy as np 
import pandas as pd
import matplotlib. pyplot as plt
from sklearn import cluster, datasets 
from sklearn.decomposition  import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn import metrics 

#载入iris数据集
def load_iris():
    iris = datasets.load_iris()
    X = np.array(iris.data)
    y = iris.target
    
    conn=pymysql.connect('localhost','root','9705165')
    conn.select_db('hdzz')
    cur=conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS `iris`(`id` INT(10),`item1` float,`item2` float,`item3` float,`item4` float,`target` int);")
    for i in range(len(X)):
        cur.execute("INSERT INTO iris VALUES (%d, %f, %f, %f, %f, %d);"%(i,X[i][0],X[i][1],X[i][2],X[i][3],y[i]))
        print('ok')
    cur.close()
    conn.commit()
    conn.close()

def load_number():
    n_samples = 500# 产生的样本点数目为500个
    #模拟数据
    #n_samples样本数，n_features特征数，n_informative生成输出的特征数量，noise应用于输出的高斯噪声的标准差，coef如果为真，则返回基础线性模型的系数
    #random_state确定数据集创建的随机数生成。跨多个函数调用传递可重复输出的int
    x, y, coef = datasets.make_regression(n_samples=n_samples, 
                                          n_features=1,
                                          n_informative=1, 
                                          noise=10,coef=True,
                                          random_state=0)
        

    n_outliers = 100# 前100个设为异常点
    # 添加异常数据
    np.random.seed(0)#用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同
    x[:n_outliers] = 4 + 0.5 * np.random.normal(size=(n_outliers, 1))
    y[:n_outliers] = -20 + 20 * np.random.normal(size=n_outliers)
    conn=pymysql.connect('localhost','root','9705165')
    conn.select_db('hdzz')
    cur=conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS `number`(`id` INT(10),`item1` float,`item2` float);")
    for i in range(len(y)):
        cur.execute("INSERT INTO number VALUES (%d, %f, %f);"%(i,x[i][0].item(),y[i].item()))
        print('ok')
    cur.close()
    conn.commit()
    conn.close()


