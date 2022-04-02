import cv2

handDetect = cv2.CascadeClassifier('Hand.Cascade.1.xml')

stream = cv2.VideoCapture(0)

while True:
	success, img = stream.read()
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hand = handDetect.detectMultiScale(gray, 1.3, 5)

	for (x, y, w, h) in hand:
		img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

	cv2.imshow('Live', img)

	if cv2.waitKey(10) & 0xFF == ord('\x1b'):
		break
