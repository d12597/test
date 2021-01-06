from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_score,  recall_score, f1_score
import matplotlib.pyplot as plt  
iris = load_iris()
#print(iris)
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
etree_clf = DecisionTreeClassifier(criterion='entropy')
gtree_clf = DecisionTreeClassifier(criterion='gini')
etree_clf.fit(x_train, y_train)
gtree_clf.fit(x_train, y_train)
export_graphviz( 
        gtree_clf,
        out_file="C:/Users/zxcv7/Desktop/iris_tree.dot",
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

gscores = cross_val_score(gtree_clf, x_test, y_test, cv=5) 
escores = cross_val_score(etree_clf, x_test, y_test, cv=5) 
print('gnin:',gscores) 
print('entropy',escores) 

gy_test_pred = gtree_clf.predict(x_test)
gtest_precision = precision_score(y_test, gy_test_pred,average='micro')
gtest_recall = recall_score(y_test, gy_test_pred,average='micro')
gtest_f1 = f1_score(y_test, gy_test_pred,average='micro')

ey_test_pred = etree_clf.predict(x_test)
etest_precision = precision_score(y_test,ey_test_pred,average='micro')
etest_recall = recall_score(y_test, ey_test_pred,average='micro')
etest_f1 = f1_score(y_test, ey_test_pred,average='micro')
l1=['ginitp','entropytp','entropyre','ginire','ginif1','entropyf1']
l2=[]
l2.append(gtest_precision)
l2.append(etest_precision)
l2.append(gtest_recall)
l2.append(etest_recall)
l2.append(gtest_f1)
l2.append(etest_f1)
print(l2)
plt.bar(range(len(l1)), l2,tick_label=l1)  
