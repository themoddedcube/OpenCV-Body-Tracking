import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

stream = cv2.VideoCapture(0)       

while True:
	success, img = stream.read()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.3, 3)

	for (fx, fy, fw, fh) in faces:
		cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 1)
		cv2.putText(img, 'face', (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

	cv2.imshow('Live', img)

	if cv2.waitKey(10) & 0xFF == ord('\x1b'):
		break

stream.release()
cv2.destroyAllWindows()