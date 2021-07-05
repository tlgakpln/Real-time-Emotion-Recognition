import sys, os
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, LeakyReLU
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

# df=pd.read_csv('fer2013.csv')
# df2=pd.read_csv('train_2.csv')
df = pd.read_csv('lastdataset1.csv')

# print(df.info())
# print(df["Usage"].value_counts())

# print(df.head())
# X_train,train_y=[],[]

X = df.drop(['emotion'], axis=1)
y = df['emotion']

X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(X, y, np.arange(len(X)), test_size=0.3, stratify=y, random_state=15)

# for index, row in df.iterrows():
#     val=row['pixels'].split(" ")
#     try:
#         if 'Training' in row['Usage']:
#            X_train.append(np.array(val,'float32'))
#            train_y.append(row['emotion'])
#         elif 'PublicTest' in row['Usage']:
#            X_test.append(np.array(val,'float32'))
#            test_y.append(row['emotion'])
#     except:
#         print(f"error occured at index :{index} and row:{row}")


# for index, row in df.iterrows():
#     val=row['pixels'].split(" ")
#     try:
#        X_train.append(np.array(val,'float32'))
#        train_y.append(row['emotion'])
#
#     except:
#         print(f"error occured at index :{index} and row:{row}")


num_features = 64
num_labels = 7
batch_size = 64
epochs = 1000
width, height = 48, 48

# df_train=np.array(df2.drop(['emotion'],axis=1),'float32')
X_train = np.array(X_train, 'float32')
y_train = np.array(y_train)
# X_train= np.concatenate(X_train,np.array(df2.drop(['emotion'],axis=1),'float32'))
# train_y= np.concatenate((train_y,np.array(df2['emotion'],'float32')))
X_test = np.array(X_test, 'float32')
y_test = np.array(y_test)

y_train = np_utils.to_categorical(y_train, num_classes=num_labels)
y_test = np_utils.to_categorical(y_test, num_classes=num_labels)

# cannot produce
# normalizing data between oand 1
# X_train -= np.mean(X_train, axis=0)
# X_train /= np.std(X_train, axis=0)
#
# X_test -= np.mean(X_test, axis=0)
# X_test /= np.std(X_test, axis=0)

X_train /= 255
X_test /= 255

X_train_ = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test_ = X_test.reshape(X_test.shape[0], 48, 48, 1)

# print(f"shape:{X_train.shape}")
##designing the cnn
# 1st convolution layer
# model = Sequential()
#
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
# model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
# # model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
# model.add(Dropout(0.5))
#
# #2nd convolution layer
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# # model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
# model.add(Dropout(0.5))
#
# #3rd convolution layer
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# # model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
#
# model.add(Flatten())
#
# #fully connected neural networks
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))

model = Sequential()
model.add(Conv2D(64, 4, 4, padding='same', input_shape=(X_train_.shape[1:])))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(0.1))
model.add(Dropout(0.25))

model.add(Conv2D(128, 4, 4, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(0.1))
model.add(Dropout(0.25))

model.add(Conv2D(256, 4, 4, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(0.1))
model.add(Dropout(0.25))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(256))
model.add(BatchNormalization())
model.add(LeakyReLU(0.1))
model.add(Dropout(0.25))
model.add(Dense(num_labels, activation='softmax'))

model.summary()

# Compliling the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])
model = load_model('model/model2.h5')
# Training the model
model.fit(X_train_, y_train, batch_size=batch_size, epochs=2000,
          validation_data=(X_test_, y_test), shuffle=True, verbose=1)
model.save('model/model2.h5')



y_head = model.predict(X_test_)
y_train_head = model.predict(X_train_)

# column_names=[]
# for i in range(X_train.shape[1]):
#     column_names.append('pxl'+str(i))
# X_train=pd.DataFrame(X_train,columns=column_names)
# X_test=pd.DataFrame(X_test,columns=column_names)
# X_train.index=idx1
# X_test.index=idx2
#
# y_head[y_head > 0.7] = 1
# y_head[y_head < 0.7] = 0
# y_train_head[y_train_head > 0.7] = 1
# y_train_head[y_train_head < 0.7] = 0
#
# label_names=[]
# for i in range(y_train.shape[1]):
#     label_names.append('label'+str(i))
# y_train=pd.DataFrame(y_train,columns=label_names)
# y_train_head=pd.DataFrame(y_train_head,columns=label_names)
# y_test=pd.DataFrame(y_test,columns=label_names)
# y_head=pd.DataFrame(y_head,columns=label_names)
# y_train.index=idx1
# y_train_head.index=idx1
# y_test.index=idx2
# y_head.index=idx2
#
# result_train = y_train == y_train_head
# result_test= y_test == y_head
#
# plt.style.use('seaborn-white')
# y_head_class=np.argmax(y_head,axis=1)
# Y_true=np.argmax(y_test,axis=1)
# conf_mtrx=confusion_matrix(Y_true,y_head_class)
# plt.subplots(figsize=(20,15))
# sb.heatmap(conf_mtrx,annot=True, annot_kws={"size": 16}, linewidths=0.8,cmap="viridis",fmt='d', linecolor="purple")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix", fontsize=25, color='indigo')
# plt.show()


# Saving the  model to  use it later on
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
