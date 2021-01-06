# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:03:40 2020

@author: Vonco
"""


import numpy as np
import matplotlib.pyplot as plt #绘图库
from sklearn.linear_model import LinearRegression #线性回归算法
from sklearn import linear_model, datasets #引入线性回归模型
from sklearn.model_selection import train_test_split#用于将矩阵随机划分为训练子集和测试子集
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score#求R2系数
from sklearn.metrics import mean_squared_error#平方误差
# from sklearn import  metrics

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False#用来正常显示负号

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


plt.plot(x[n_outliers:], y[n_outliers:],'go')
plt.plot(x[:n_outliers], y[:n_outliers], 'yo')
plt.show()


#创建模型
liner = LinearRegression()
#拟合模型
liner.fit(np.reshape(x,(-1,1)),np.reshape(y,(-1,1)))
print(liner)
# 拟合直线
lr = linear_model.LinearRegression()#实例化一个线性回归的模型
lr.fit(x, y)#在x,y上训练一个线性回归模型，如果训练顺利，则regr会存储训练完成之后的结果模型。

# 计算R2以及MSE
y_pred = lr.predict(x)
r2 = r2_score(y,y_pred)
print("线性回归:r2 = %f"%r2)

#计算预测值和实际值的平均平方误差
MSE = mean_squared_error(y,y_pred)
print("线性回归:MSE=%f\n"%MSE)


# 用RANSAC算法对线性模型进行拟合  稳健回归拟合
ransac = linear_model.RANSACRegressor(base_estimator = linear_model.LinearRegression(),min_samples = 10,residual_threshold = 25.0,stop_n_inliers = 300,max_trials = 100,random_state = 0)
ransac.fit(x, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)#逻辑非，这里指与inlier_mask相反，返回bool类型

# 估计模型的预测数据
line_x = np.arange(x.min(), x.max())[:, np.newaxis]
line_y = lr.predict(line_x)
line_y_ransac = ransac.predict(line_x)

lw = 3
plt.plot(x[n_outliers:], y[n_outliers:],'go')
plt.plot(x[:n_outliers], y[:n_outliers], 'yo')
plt.plot(line_x, line_y, color='navy', linewidth=lw, label='线性回归')
plt.plot(line_x, line_y_ransac, color='r', linewidth=lw,label='RANSAC regressor')
plt.show()



#RANCE  R2系数
y_pred = lr.predict(line_x)
r2 = r2_score(line_y,y_pred)
print("RANCE:r2 = %f"%r2)

#计算预测值和实际值的平均平方误差
MSE = mean_squared_error(line_y,y_pred)
print("RANCE:MSF=%f\n"%MSE)

#划分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)


# 建立线性回归，并用训练的模型绘图
#一元线性回归
liner.fit(X_test, y_test)
xx = np.linspace(-3, 5, 100)
yy = liner.predict(xx.reshape(xx.shape[0], 1))
plt.plot(X_test, y_test, '*')
#plt.plot(X_train, y_train, 'k.')
plt.plot(xx, yy)
plt.plot(xx, yy, color='g', linewidth=lw)
plt.show()


y_test_pred = lr.predict(X_test)
r2 = r2_score(y_test,y_test_pred)
print("一元线性回归:r2 = %f"%r2)

#计算预测值和实际值的平均平方误差
MSE = mean_squared_error(y_test,y_test_pred)
print("一元线性回归:MSF=%f\n"%MSE)

#plt = runplt()

#一元二次线性回归
quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)
liner_quadratic = LinearRegression()
liner_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, liner_quadratic.predict(xx_quadratic), 'c-')

plt.plot(X_test[n_outliers:], y_test[n_outliers:],'go')
plt.plot(X_test[:n_outliers], y_test[:n_outliers],  '*')
plt.plot(xx, liner_quadratic.predict(xx_quadratic), color='c', linewidth=lw)
plt.show()

print('一元二次线性回归：R2=', liner_quadratic.score(X_test_quadratic, y_test))

# y_test_pred = lr.predict(quadratic_featurizer)
# MSE = metrics.mean_squared_error(y_test, y_test_pred)
# print("一元二次线性回归：MSF=%f\n"%MSE)



# 比较估计系数
print("斜率：",coef, "斜率：",lr.coef_, "RANSAC算法参数： ",ransac.estimator_.coef_)



lw = 3 #线条的粗细
#预测
y_pred = liner.predict(np.reshape(x,(-1,1)))
plt.scatter(x[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',label='内点')
plt.scatter(x[outlier_mask], y[outlier_mask], color='gold', marker='.',label='噪声点')

#一元线性回归
plt.plot(X_test, y_test, '*')
# plt.plot(X_train, y_train, 'k.')
plt.plot(xx, yy)
plt.plot(xx, yy, color='g', linewidth=lw, label='一元线性回归')


#一元二次线性回归
plt.plot(xx, liner_quadratic.predict(xx_quadratic), 'c-')
plt.plot(xx, liner_quadratic.predict(xx_quadratic), color='c', linewidth=lw, label='一元二次线性回归')


plt.plot(line_x, line_y, color='navy', linewidth=lw, label='线性回归')
plt.plot(line_x, line_y_ransac, color='r', linewidth=lw,label='RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel("输入")
plt.ylabel("响应")
plt.title("回归模型")
plt.show()#显示图像
