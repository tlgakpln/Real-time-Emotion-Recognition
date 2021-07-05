import pandas as pd
import cv2
import glob




emotions=['angry', 'disgust','fear', 'happy', 'sad', 'surprise', 'neutral']


def create_emotion_datasets(folder_name):
    img_list=[]
    for img in glob.glob('train_ek/'+folder_name+'/' + folder_name +'/*.png'):
        image= cv2.imread(img,0)
        flatten_img=image.flatten()
        img_list.append(flatten_img)
    img_dataset=pd.DataFrame(img_list)

    if folder_name=='angry':
        img_dataset['emotion']=0
    elif folder_name=='disgust':
        img_dataset['emotion']=1
    elif folder_name=='fear':
        img_dataset['emotion']=2
    elif folder_name=='happy':
        img_dataset['emotion']=2
    elif folder_name=='sad':
        img_dataset['emotion']=3
    elif folder_name=='surprise':
        img_dataset['emotion']=4
    elif folder_name=='neutral':
        img_dataset['emotion']=5


    return img_dataset
new_train=[]
for i in emotions:
    dataset=create_emotion_datasets(i)
    new_train.append(dataset)
df=pd.concat(new_train)
df.reset_index(drop=True,inplace=True)
df.to_csv('last_dataset2.csv',index=False)