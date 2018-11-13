# coding = gbk
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras import backend as K
K.set_image_dim_ordering('tf')  
MODEL_FILENAME = "/home/zwh/tensor-test/lstm_model_kdd.hdf5"

path = '/home/zwh/tensor-test/kddtrain.csv'
df=pd.read_csv(path,header=None)
if df.empty:
   print("Data is empty")
print(df.shape) #(1048576, 42)
X = df.iloc[0:df.shape[0],0:41].values #x-data
Y = df.iloc[0:df.shape[0],41].values #y-label
#print(X.shape) #(1048576, 41)
#print(Y.shape) #(1048576,)

labelencoder_x = LabelEncoder()
X[:, 1] = labelencoder_x.fit_transform(X[:, 1])
X[:, 2] = labelencoder_x.fit_transform(X[:, 2])
X[:, 3] = labelencoder_x.fit_transform(X[:, 3])
Dict = {}
for inlist in Y:
	if Dict.get(inlist) == None:
		Dict[inlist] = 1
	else:
		Dict[inlist] += 1
List = list(Dict.keys())
num = len(List)
#print(num) #20

X = np.expand_dims(X, axis=2)
X = np.array(X)
Y = np.array(Y)
X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.20, random_state = 36)

lb = LabelEncoder().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_valid = lb.transform(Y_valid)
in_Y_train = np_utils.to_categorical(Y_train)
in_Y_valid = np_utils.to_categorical(Y_valid)
# print(np.shape(in_Y_train))


# # LSTM with Dropout and CNN classification
# max_feature = max(df.iloc[0:df.shape[0],4].values)
# embedding_vector_length= 1
# max_review_length = 41
model=Sequential()
# model.add(Embedding(max_feature+1,embedding_vector_length,input_length=max_review_length))
# model.add(SpatialDropout1D(0.3))

model.add(Conv1D(activation="relu", padding="same", filters=64, kernel_size=5, input_shape=(41,1)))

model.add(MaxPooling1D(pool_size=4))

model.add(LSTM(100))

model.add(Dense(num,activation="softmax"))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

model.fit(X_train,in_Y_train,validation_data=(X_valid,in_Y_valid), epochs=10,batch_size=64)

model.save(MODEL_FILENAME)

preds = model.predict(X_valid)
# preds = np.argmax(preds_test,axis=1)
print('preds:',preds)

letter = preds.argmax(axis=1)
print('predicted:',letter)

Y_valid = in_Y_valid.argmax(axis=1)
print('Y_valid:',Y_valid)

num = 0.0
sum = len(preds)
print('sum:',sum)
for i in range(sum):
    if letter[i] == Y_valid[i]:
        num += 1
print('num:',num)
print(num/sum)