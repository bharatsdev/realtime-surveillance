import numpy as np
import imutils
import cv2


class SingleMotionDetector:
    def __init__(self, accumWeight=0.5):
        # Store the Accumulated weight factor
        self.accumWeight = accumWeight
        # Initialize the background model
        self.bg = None

    def update(self, image):
        # If the background model is None, Initialize it
        if self.bg is None:
            self.bg = image.copy().astype('float')

        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    def detect(self, image, tVal=25):
        # Compute the absolute difference between the background model
        # and the image passed in then threshold the delta image
        delta = cv2.absdiff(self.bg.astype('uint8'), image)
        thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]
        # Perform a series of erosion and dilation to remove small blobs

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find Contours in the threshold image
        # and initialize the minimum and maximum bounding box regions for motion
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

        # If No Contours where found, Return None
        if len(cnts) == 0:
            return None
        # Otherwise, loop over the contours
        for cnt in cnts:
            # Compute the bounding box of the contour and use
            #  it fo update the minimum and maximum bounding box regions
            (x, y, w, h) = cv2.boundingRect(cnt)
            (minX, minY) = (min(minX, x), min(minY, y))
            (maxX, maxY) = (max(maxX, (x + w)), max(maxY, (y + h)))

        # Otherwise, return a tuple of the thresholded image along with bounding box
        return (thresh, (minX, minY, maxX, maxY))
