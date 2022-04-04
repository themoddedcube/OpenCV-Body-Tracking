import cv2

handCascade = cv2.CascadeClassifier('haarcascade_fist.xml')

stream = cv2.VideoCapture('vid1.mp4')
#stream = cv2.VideoCapture(0)                              #Uncomment to use webcam as videocapture device

while True:
	success, img = stream.read()
	img = cv2.resize(img, (1080, 720))                    #Comment when using webcam as videocapture device for best results
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hand = handCascade.detectMultiScale(gray, 1.3, 5)

	for (hx, hy, hw, hh) in hand:
		cv2.rectangle(img, (hx, hy), (hx + hw, hy + hh), (255, 0, 0), 1)
		cv2.putText(img, 'fist', (hx, hy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

	cv2.imshow('Live', img)

	if cv2.waitKey(10) & 0xFF == ord('\x1b'):
		break

stream.release()
cv2.destroyAllWindows()
