import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

stream = cv2.VideoCapture(0)

while True:
	success, img = stream.read()
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.3, 3)
	eye = eyeCascade.detectMultiScale(gray, 1.3, 6)

	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
		cv2.putText(img, 'face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

	for (ex, ey, ew, eh) in eye:
		cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)
		cv2.putText(img, 'eye', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

	cv2.imshow('Live', img)

	if cv2.waitKey(10) & 0xFF == ord('\x1b'):
		break

stream.release()
cv2.destroyAllWindows()


