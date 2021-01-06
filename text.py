# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:34:08 2021

@author: 80686
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import pymysql
mpl.rcParams['font.sans-serif'] = ['SimHei']

directory = 'C:/Users/80686/Desktop/数据挖掘大实验/20news-18828'
category_names = os.listdir(directory)
labels = list()
content=list()
p=[]
r=[]
f1=[]
n=[]
a=[]
b=[]
c=[]
f=[]
content=[]

for i in range(len(category_names)):
    category = category_names[i]
    category_dir = os.path.join(directory, category)
    for file_name in os.listdir(category_dir):
        f.append(file_name)
        file_path = os.path.join(category_dir,file_name)
        contents = open(file_path, encoding='latin1').read().strip()
        labels.append(i + 1)
        content.append(contents[:50].replace("\n",""))
tfidf_vectorizer = TfidfVectorizer()

train_contents, test_contents, train_labels, test_labels = \
train_test_split(content, labels, shuffle=True, test_size=0.2)

model_name=["lr","svm","mul"]
svm= LinearSVC(verbose=True)
mul= MultinomialNB()
lr= LogisticRegression()
chi2_feature_selector = SelectKBest(chi2, k=10000)


def classifaction_report_csv(report):
    lines = report.split('\n')
    for line in lines[2:-5]:
        row_data = line.split()
        p.append(float(row_data[1]))
        r.append(float(row_data[2]))
        f1.append(float(row_data[3]))
        if len(n)<20:
            n.append(row_data[0])

def muti_score(model):
    pipeline = Pipeline(memory=None, steps=[
    ('tfidf', tfidf_vectorizer),
    ('chi2', chi2_feature_selector),
    ('model', model),
    ])
    pipeline.fit(train_contents, train_labels)
    result = pipeline.predict(test_contents)
    report = classification_report(test_labels, result, target_names=category_names)
    classifaction_report_csv(report)
for name in model_name:
    model=eval(name)
    muti_score(model)


def draw():
    for i in range(20):
        a.append([p[i],p[i+20],p[i+40]])
        b.append([r[i],r[i+20],r[i+40]])
        c.append([f1[i],f1[i+20],f1[i+40]])
    data1=pd.DataFrame(a,columns=['lr','svm','mul'],index=n)
    data1.plot(kind='bar',
          figsize=(20,5),
          title='precision对比柱状图')
    data2=pd.DataFrame(b,columns=['lr','svm','mul'],index=n)
    data2.plot(kind='bar',
          figsize=(20,5),
          title='recall对比柱状图')
    data3=pd.DataFrame(c,columns=['lr','svm','mul'],index=n)
    data3.plot(kind='bar',
          figsize=(20,5),
          title='F1-measure对比柱状图')

draw()
conn=pymysql.connect('localhost','root','9705165')
conn.select_db('hdzz')
cur=conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS `text`(`id` INT(10),`item1` varchar(255),`item2` varchar(255));")
print(f)
print(content)
for i in range(len(f)):
    cur.execute("INSERT INTO text VALUES (%d, %s, %s);"%(i,f[i],content[i]))
    print('ok')
cur.close()
conn.commit()
conn.close()
