import enum
import cv2
import mediapipe as mp
import time 

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

stream = cv2.VideoCapture('vid1.mp4')
prev_frame_time = 0
new_frame_time = 0

while True:
    success, img = stream.read()
    img = cv2.resize(img, (771, 480))

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img)
    #print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS, mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))            

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.putText(img, str(int(fps)) + ' fps', (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 0), 3)
    cv2.imshow('vid', img)

    if cv2.waitKey(1) & 0xFF == ord('\x1b'):
        break

stream.release()
cv2.destroyAllWindows()
