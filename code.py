import cv2
import numpy as np

im_gray = cv2.imread("sapviii.bmp",0)

thresh = 127
im_binerr = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)[1]

im_gray = cv2.medianBlur(im_gray,5)
im_biner = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(im_gray, cv2.HOUGH_GRADIENT,1,20,param1=290, param2=55,
                           minRadius=0, maxRadius=0)
circles = np.uint16(np.around(circles))

# cv2.imshow("hasil", circles)
for i in circles[0,:]:
    cv2.circle(im_biner, (i[0],i[1]), i[2], (0,255,255), 2)
    cv2.circle(im_biner, (i[0],i[1]), 2, (0,0,255), 112)

flag=1
row, col, ch = im_biner.shape
graykanvas = np.zeros((row, col, 1), np.uint8)
for i in range(0, row):
    for j in range(0, col):
        b, g, r = im_biner[i, j]
        if(b == 255 & g==0 & r==0):
            graykanvas.itemset((i,j,0), 255)
            if(flag==1):
                x=i
                y=j
                flag = 100
        else:
            graykanvas.itemset((i,j,0), 0)


im_hasil = cv2.subtract(graykanvas, im_gray)


hasil_crop = im_hasil[x:x+112, y-56:y+56] #im awe [y,x]

cv2.imshow("hasil2", hasil_crop)
# cv2.imshow("hasil", im_biner)
cv2.waitKey()
