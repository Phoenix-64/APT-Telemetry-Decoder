import cv2

img = cv2.imread("noaa18.png", 0)
img_c = img[111:875]
cv2.imshow("Cropted", img_c, )
cv2.waitKey(0)

cv2.imwrite("noaa18c.png", img_c)