import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
# 计算机实现的随机数生成通常为伪随机数生成器，为了使得具备随机性的代码最终的结果可复现，需要设置相同的种子值
rng = np.random.RandomState(42)
df=pd.read_csv('iforest.csv')
if df.empty:
   print("Data is empty")
df=df.sample(frac=1,random_state=rng)
X_data = df.iloc[0:,0:41].values #获取自变量
Y_data = df.iloc[0:,41].values  #获取因变量，即标签

#print(X_data)
print(Y_data)
#print(X_data[0, 1:4])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer

labelencoder_x = LabelEncoder()
X_data[:, 1] = labelencoder_x.fit_transform(X_data[:, 1])
X_data[:, 2] = labelencoder_x.fit_transform(X_data[:, 2])
X_data[:, 3] = labelencoder_x.fit_transform(X_data[:, 3])
#print(X_data[0, 1:4])
#print(data[0, 1:4])
labelencoder_y = LabelEncoder()
Y_data = labelencoder_y.fit_transform(Y_data)
print(Y_data)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_data,Y_data, test_size = 0.50, random_state = rng)
# fit the model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_valid)
for i in range(len(y_pred_train)):
    if(y_pred_train[i]==-1):
        y_pred_train[i]+=1
print(y_pred_train)
print(Y_valid)
acc=0
for i in range(len(Y_valid)):
    if(y_pred_train[i] + Y_valid[i]==1):
        acc+=1

print(float(acc/len(Y_valid)))        


