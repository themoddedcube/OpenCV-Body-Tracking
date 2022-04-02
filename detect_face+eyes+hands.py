import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
handDetect = cv2.CascadeClassifier('Hand.Cascade.1.xml')

stream = cv2.VideoCapture(0)


while True:
	success, img = stream.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.3, 5)
	hand = handDetect.detectMultiScale(gray, 1.3, 5)

	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		eyes = eyeCascade.detectMultiScale(roi_gray)
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255 ), 2)

	for (hx, hy, hw, hh) in hand:
		cv2.rectangle(img, (hx, hy), (hx + hw, hy + hh), (255, 0, 0 ), 2)

	cv2.imshow('Live', img)
	if cv2.waitKey(10) & 0xFF == ord('\x1b'):
		break

stream.release()
cv2.destroyAllWindows()


