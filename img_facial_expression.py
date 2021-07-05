import cv2
import numpy as np
from keras.models import model_from_json, load_model
from keras.preprocessing import image
import glob


# model = model_from_json(open("fer.json", "r").read())

model = load_model('model/model2.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

folder_name = 'fear'

for img in glob.glob('image_test/' + folder_name + '/*.png'):
    img = cv2.imread(img)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(img, (x, y), (x + w, y + h), (125, 0, 255), thickness=2)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(img, predicted_emotion, (int(x + 5), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 0, 255), 2)

    resized_img = cv2.resize(img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
