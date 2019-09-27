import cv2
img = cv2.imread('havy.jpg', 0)
cv2.line(img, (0,0), (511,511),(255,0,0), 5)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()