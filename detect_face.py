import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

stream = cv2.VideoCapture(0)

while True:
	success, img = stream.read()
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.3, 5)

	for (x, y, w, h) in faces:
		img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imshow('Live', img)

	if cv2.waitKey(10) & 0xFF == ord('\x1b'):
		break

