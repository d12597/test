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
iris = datasets.load_iris()
X = np.array(iris.data)
y = iris.target
# print(X)

# # 将4维特征使用箱图进行可视化
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"  
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']  
# dataset = pd.read_csv(url, names=names)
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

#使用PCA将数据降为2维
pca = PCA(n_components = 2) #降为2维
pca = pca.fit(X)
X_dr = pca.transform(X)
##print(X_dr)
#print(X_dr[y == 0,0])
plt.scatter(X_dr[y==0,0],X_dr[y==0,1],c="red",label=iris.target_names[0]) #山鸢尾
plt.scatter(X_dr[y==1,0],X_dr[y==1,1],c="blue",label=iris.target_names[1])#变色鸢尾
plt.scatter(X_dr[y==2,0],X_dr[y==2,1],c="black",label=iris.target_names[2])#维吉尼亚鸢尾
plt.legend() #给图像加上图例
plt.title("PCA of iris dataset")
plt.show()

print(pca.explained_variance_)#两个主成分可解释方差大小
print(pca.explained_variance_ratio_)#主成分占比
print((pca.explained_variance_ratio_).sum())#第一，二主成分共携带97.77%的信息

pca_line = PCA().fit(X)
plt.plot([1,2,3,4],np.cumsum(pca_line.explained_variance_ratio_))
plt.xticks([1,2,3,4])#累计贡献率曲线

#聚类种类及名称
clustering_names = ['MiniBatchKMeans',  'MeanShift', 'AgglomerativeClustering','DBSCAN', 'Birch']
#设置fugure宽高，单位：英寸
plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
#调整图片位置及间距
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,hspace=.01)

plot_num = 1
x = X_dr 
#规范化数据集以便于参数选择
x = StandardScaler().fit_transform(x)
#均值漂移估计带宽
bandwidth = cluster.estimate_bandwidth(x, quantile=0.3)    
#kneighbors_graph类返回用KNN时和每个样本最近的K个训练集样本的位置
connectivity = kneighbors_graph(x, n_neighbors=10, include_self=False)    
#使连接对称 
connectivity = 0.5 * (connectivity + connectivity.T)

# 创建聚类估计器

two_means = cluster.MiniBatchKMeans(n_clusters=3,n_init=10)     #MiniBatchKMeans
ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)   #MeanShift
average_linkage = cluster.AgglomerativeClustering(n_clusters=3) #AgglomerativeClustering
dbscan = cluster.DBSCAN(eps=0.5)                                #DBSCAN
birch = cluster.Birch(n_clusters=3)                             #Birch

#聚类算法
clustering_algorithms = [two_means,ms,average_linkage,dbscan, birch]

colors = np.array([x for x in  "bgrcmykbgrcmykbgrcmykbgrcmyk"])
#hstack()函数水平把数组堆叠起来
colors = np.hstack([colors] * 20)

num = []
for name, algorithm in zip(clustering_names, clustering_algorithms):
   
    t0 = time.time() #time()函数返回当前时间的时间戳
    algorithm.fit(X)
    t1 = time.time()
    #hasattr（）函数用于判断对象是否包含对应的属性
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(x)

    # plot
    plt.subplot(4, len(clustering_algorithms), plot_num)

    plt.title(name, size=18)
    plt.scatter(x[:, 0], x[:, 1], color=colors[y_pred].tolist(), s=10)

    if hasattr(algorithm, 'cluster_centers_'):
        centers = algorithm.cluster_centers_
        center_colors = colors[:len(centers)]
        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
    #设置x坐标轴范围
    plt.xlim(-2, 2)
    #设置y坐标轴范围
    plt.ylim(-2, 2)
    #设置x轴
    plt.xticks(())
    #设置y轴
    plt.yticks(())
    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
    transform=plt.gca().transAxes, size=15,
    horizontalalignment='right')  #添加标签
    
    num.append(metrics.v_measure_score(y,y_pred))
    plot_num += 1
    
#设置不同K值
s = []
for k in range(2,10):
    km = cluster.MiniBatchKMeans(n_clusters=k)
    km.fit(x)
    s.append(metrics.v_measure_score(y,km.predict(x)))
    
#折线图
plt.figure("1",figsize=(10,7))
a=[2,3,4,5,6,7,8,9]
plt.title("MiniBatchKMeans")
plt.plot(a,s,"b",linewidth=3)
plt.xlabel("k")
plt.ylabel("v_measure_score")
plt.show()

#直方图
plt.figure("2",figsize=(10,7))
plt.title("v_measure_score")
plt.bar(range(len(num)),num,color="g",tick_label=clustering_names)
plt.show()


    
