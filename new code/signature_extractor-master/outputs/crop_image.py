import numpy as np
import cv2

img = cv2.imread('output.png') # Read in the image and convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
coords = cv2.findNonZero(gray) # Find all non-zero points (text)
x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
rect = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image
cv2.imshow("Cropped", rect) # Show it
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Output_Crop.png", rect) # Save the image