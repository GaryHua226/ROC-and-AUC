# -*- coding: utf-8 -*-
import numpy as np

############计算ROC需要的库函数#############
from sklearn import metrics
import matplotlib.pyplot as plt

#############计算fpr,tpr##################
##y是一个一维数组（样本的真实分类），数组值表示类别（一共有两类，1和2），人工标注，属于测试集的真实分类
##score即各个样本属于正例的概率；是网络的输出；首先用训练集训练网络，然后利用测试集的数据产生的
##fpr, tpr是ROC的横纵坐标
##thresholds是截断阈值
y = np.array([1, 0, 1, 1, 1, 0, 0, 0])
# scores = np.array([0.6, 0.31, 0.58, 0.22, 0.4, 0.51, 0.2, 0.33])
scores = np.array([0.04, 0.1, 0.68, 0.24, 0.32, 0.12, 0.8, 0.51])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
print(fpr)
print(tpr)
AUC = 0
for i in range(len(fpr)-1):
    x = fpr[i+1] - fpr[i]
    y = tpr[i]
    AUC += x * y
print('AUC: ', AUC)
#############画图##################
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fpr, tpr, '--*b')

plt.show()