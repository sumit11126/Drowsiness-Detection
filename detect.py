import cv2
import dlib
from scipy.spatial import distance
import time


def calculate_eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio


cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")

eyes_closed_time = None

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    eyes_open = False

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_eye_aspect_ratio(leftEye)
        right_ear = calculate_eye_aspect_ratio(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR, 2)

        if EAR > 0.20:  # Eyes are open
            eyes_open = True

    if eyes_open:
        eyes_closed_time = None
    elif eyes_closed_time is None:
        eyes_closed_time = time.time()
        print("Eyes closed")
    else:
        if time.time() - eyes_closed_time > 2:  # Change this value to change the time
            cv2.putText(frame, "Eyes Closed for 3+ seconds", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            print("Eyes closed for 3+ seconds")
    print('Eyes closed',eyes_closed_time )
    cv2.imshow("Drowsiness Detection", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
