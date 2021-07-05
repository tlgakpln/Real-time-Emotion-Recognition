import cv2
import glob


def crop_face(folder_name):
    for img in glob.glob('train_ek/' + folder_name +'/*.png'):


        image = cv2.imread(img, 0)
        face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_haar_cascade.detectMultiScale(image, 1.25, 6)
        # Print number of faces found
        print('Number of faces detected:', len(faces))
        image_copy = image.copy()
        # Get the bounding box for each detected face
        num = 0
        for f in faces:
            x, y, w, h = [v for v in f]
            # cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 0), 3)
            face_crop = image_copy[y:y + h, x:x + w]
            img_resize=cv2.resize(face_crop, (48, 48))
            cv2.imwrite('train_ek/' + folder_name + '/' + folder_name + '/' +str(img[-8:-3])+ str(num) +'.png', img_resize)
            num += 1

    return face_crop


face_crop = crop_face('surprise')

# image = cv2.imread('train_ek/disgust/adsiz.png', 0)
# face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# faces = face_haar_cascade.detectMultiScale(image, 1.3, 8)
# # Print number of faces found
# print('Number of faces detected:', len(faces))
# image_copy = image.copy()
# # Get the bounding box for each detected face
# num = 0
# for f in faces:
#     x, y, w, h = [v for v in f]
#     cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     face_crop = image_copy[y:y + h, x:x + w]
#
#     cv2.imwrite('train_ek/'+'disgust'+'/'+'disgust'+'/'+str(num)+'.png', face_crop)
#     num += 1
# cv2.imshow('image',image_copy)
# cv2.waitKey(0)