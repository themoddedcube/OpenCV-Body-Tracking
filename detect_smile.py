import cv2

smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")

stream = cv2.VideoCapture(1)

while True:
	success, img = stream.read()
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	smile = smileCascade.detectMultiScale(gray, 1.3, 21)

	for (sx, sy, sw, sh) in smile:
		cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
		cv2.putText(img, 'smile', (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

	cv2.imshow('Live', img)

	if cv2.waitKey(10) & 0xFF == ord('\x1b'):
		break

