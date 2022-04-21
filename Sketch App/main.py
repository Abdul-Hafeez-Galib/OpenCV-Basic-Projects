import cv2
import numpy as np

def sketch(image):
   grayimg=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert the captured image into gray scale image
   blurimg=cv2.GaussianBlur(grayimg, (3,3), 0) # blurring the image
   edges=cv2.Canny(blurimg, 10, 80) # extracting edges
   ret, mimg=cv2.threshold(edges,50,255,cv2.THRESH_BINARY) # applying threshold
   return mimg


vid_capt=cv2.VideoCapture(0) # Capturing video from webcam

# Capturing the video frame by frame
while True: 
   ret,pic_capt=vid_capt.read()
   cv2.imshow('Your Sketch', sketch(pic_capt))
   if cv2.waitKey(1)==13: # Key13 is ENTER_KEY
    break

vid_capt.release() # releasing_webcam
cv2.destroyAllWindows() # destroying_window