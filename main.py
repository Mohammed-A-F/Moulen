import cv2
import dlib
import numpy as np
from imutils import face_utils
import time

gender_model = cv2.dnn.readNetFromCaffe('mods/gender_deploy.prototxt', 'mods/gender_net.caffemodel')
age_model = cv2.dnn.readNetFromCaffe('mods/age_deploy.prototxt', 'mods/age_net.caffemodel')

gender_confidence_threshold = 0.5
age_confidence_threshold = 0.9

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('mods/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

total_faces = 0
total_age = 0
total_male = 0
total_female = 0

start_time = time.time()
while True:
    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        gender_model.setInput(blob)
        gender_predictions = gender_model.forward()
        age_model.setInput(blob)
        age_predictions = age_model.forward()

        gender = 'Male' if gender_predictions[0][0] > gender_predictions[0][1] else 'Female'
        age_index = np.argmax(age_predictions)
        age = age_index + 1

        text = '{}: {:.2f}%'.format(gender, max(gender_predictions[0])*100)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        age = int(age * 3.8)
        text = 'Age: {}'.format(age)
        cv2.putText(frame, text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        total_faces += 1
        total_age += age
        if gender == 'Male':
            total_male += 1
        else:
            total_female += 1

    cv2.imshow('frame', frame)

    if time.time() - start_time > 3:
        if total_faces > 0:
            avg_age = total_age / total_faces
            avg_male = total_male / total_faces
            avg_female = total_female / total_faces

            if avg_age <= 12:
                if avg_male > avg_female:
                    video_path = 'videos/kids_male_video.mp4'
                else:
                    video_path = 'videos/kids_female_video.mp4'
            else:
                if avg_male > avg_female:
                    video_path = 'videos/adult_male_video.mp4'
                else:
                    video_path = 'videos/adult_female_video.mp4'

            video = cv2.VideoCapture(video_path)

            while True:
                ret, frame = video.read()

                if not ret:
                    break

                cv2.imshow('video',frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            video.release()

        total_faces = 0
        total_age = 0
        total_male = 0
        total_female = 0
        start_time = time.time()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()