import cv2

eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

stream = cv2.VideoCapture(1)

while True:
	success, img = stream.read()
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	eye = eyeCascade.detectMultiScale(gray, 1.3, 5)

	for (ex, ey, ew, eh) in eye:
		cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)
		cv2.putText(img, 'eye', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

	cv2.imshow('Live', img)

	if cv2.waitKey(10) & 0xFF == ord('\x1b'):
		break

