import pandas as pd
import cv2
import numpy as np

dataset = pd.read_csv('fer2013.csv')


# Veri setini hazırla.
# Veri setini duygulara göre sınıflandır.
# Sınıflandırılan her görüntüyü klasörlerine kaydet.


# def image_dataset(df):
#     X_train, train_y = [], []
#     for index, row in df.iterrows():
#         val = row['pixels'].split(" ")
#         try:
#             X_train.append(np.array(val, 'float32'))
#             train_y.append(row['emotion'])
#
#         except:
#             print(f"error occured at index :{index} and row:{row}")
#     return X_train, train_y
#
#
# # def image_save():
# x_train, y_train = image_dataset(train)
# x_train = pd.DataFrame(x_train)
# y_train = pd.DataFrame(y_train, columns=['emotion'])
# dataset = pd.concat([x_train, y_train], axis=1)
# dataset.to_csv('fer2013.csv',index=False)


def save_img(df,k=None):
    dataset_ang=df[df['emotion'] == k]
    dataset_ang=dataset_ang.iloc[:,:-1]
    for row in dataset_ang.iterrows():
        idx, i = row
        img=np.array(i).reshape(48,48)
        if k == 0:
            cv2.imwrite('train/angry/' +'_'+ str(idx) +'_'+ '.png',img)
        elif k == 1:
            cv2.imwrite('train/disgust/' +'_'+ str(idx) +'_'+ '.png',img)
        elif k == 2:
            cv2.imwrite('train/fear/' +'_'+ str(idx) +'_'+ '.png',img)
        elif k == 3:
            cv2.imwrite('train/happy/' +'_'+ str(idx) +'_'+ '.png',img)
        elif k == 4:
            cv2.imwrite('train/sad/' +'_'+ str(idx) +'_'+ '.png',img)
        elif k == 5:
            cv2.imwrite('train/suprised/'+'_'+ str(idx) +'_'+ '.png',img)
        elif k == 6:
            cv2.imwrite('train/neutral/' +'_'+ str(idx) +'_'+ '.png',img)

    return dataset_ang

df_emo=save_img(dataset,k=6)