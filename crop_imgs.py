import cv2

img = cv2.imread("day_16.png", 0)
img_c = img[:1108]
cv2.imshow("Cropted", img_c, )
cv2.waitKey(0)

cv2.imwrite("day_16c.png", img_c)