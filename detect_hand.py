import cv2

handDetect = cv2.CascadeClassifier('haarcascade_hand.xml')

stream = cv2.VideoCapture(1)

while True:
	success, img = stream.read()
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hand = handDetect.detectMultiScale(gray, 1.3, 8)

	for (hx, hy, hw, hh) in hand:
		img = cv2.rectangle(img, (hx, hy), (hx + hw, hy + hh), (255, 0, 0), 1)
		cv2.putText(img, 'hand', (hx, hy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

	cv2.imshow('Live', img)

	if cv2.waitKey(10) & 0xFF == ord('\x1b'):
		break
