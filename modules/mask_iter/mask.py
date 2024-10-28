import cv2
import numpy as np
from argparse import ArgumentParser

# parser = ArgumentParser()
# parser.add_argument('-i', '--input', required=True, default='origin_mask.png',help='Path to input image')
# parser.add_argument('-m','mode',required=True,default='erode',help='Mode of operation')

mask = cv2.imread('mask.png')
kernel = np.ones((21,21),np.uint8)
mask = cv2.erode(mask,kernel)
# mask = cv2.dilate(mask, kernel)

cv2.imshow('res', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('mask.png', mask)