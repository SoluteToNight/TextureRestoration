import cv2
import numpy as np

mask = cv2.imread('origin_mask.png',cv2.IMREAD_GRAYSCALE)
kernel = np.ones((30,30),np.uint8)
eroded_mask = cv2.erode(mask,kernel,iterations=1)
rev_mask = 255 - eroded_mask

mask = cv2.imwrite('mask.png', eroded_mask)
cv2.imwrite('mask_rev.png', rev_mask)

# mask2 = cv2.erode(mask,kernel,iterations=3)
# mask = cv2.dilate(mask, kernel)
# difference_mask = cv2.compare(mask, mask2, cv2.CMP_GT)
# cv2.imshow('res', mask)

# cv2.imwrite('mask.png', difference_mask)
