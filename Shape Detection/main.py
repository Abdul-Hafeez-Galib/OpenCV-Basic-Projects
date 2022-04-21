import cv2
import imutils

# Define the ShapeDetector class
class ShapeDetector:
    #Constructor
    def __init__(self):
        # Since there is nothing to initialise, pass
        pass
    # Function to detect the shape. Takes contours as the second argument
    def detect(self , c):
        # Basically, the shape is unidentified
        shape = "unidentified"
        #Find the perimeter of the shape
        peri = cv2.arcLength(c , True)
        #Approximate the contours to get the number of vertices
        approx = cv2.approxPolyDP(c , 0.04 * peri , True)
        
        # Start detecting the shape based on the number of vertices. If the number of vertices is 3, its a triagle
        if len(approx) == 3:
            shape = "Triangle"
        #I f the number number of vertices is 4, the shape can either be a square or a rectangle
        elif len(approx) == 4:
            # Draw an approximate rectangle around the contours 
            (x , y , w , h) = cv2.boundingRect(approx)
            # Calculate the ratio of width to height
            av = w / float(h)
            # If the ratio is approximately equal to 1, then the shape is a square, else a rectangle
            shape = "Square" if av >= 0.95 and av <= 1.05 else "Rectangle"
        # If the number of vertices is 5, its a pentagon
        elif len(approx) == 5:
            shape = "Pentagon"
        # If the shape is none of the above, then its a circle
        else:
            shape = "Circle"
        # Return the shape of the object
        return shape

# Load the image to a variable
image = cv2.imread('shapes.jpg')
# Resize the image to get better approximation
resized = imutils.resize(image , width=300)
# Obtain factor by which the image is resized
ratio = image.shape[0] / float(resized.shape[0])
# Convert the resized image into Grayscale image
gray = cv2.cvtColor(resized , cv2.COLOR_BGR2GRAY)
# Apply Gaussian blur to the Grayscaled image
blurred = cv2.GaussianBlur(gray , (5, 5) , 0)
# Threshold the blurred image
thresh = cv2.threshold(blurred , 60 , 255 , cv2.THRESH_BINARY)[1]
# Find the contours of the copy of the threshold image
cnts = cv2.findContours(thresh.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# Create a ShapeDetector object
sd = ShapeDetector()

# Iterate over every point in contours
for c in cnts:
    # Find the moments of the contour
    M = cv2.moments(c)
    # Obtain the x and y co-ordinates of the centre of the contour
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    # Detect the shape using the contour
    shape = sd.detect(c)
    # Convert c to float
    c = c.astype("float")
    # Multiply the contour by the ratio to get the co-ordinates of the original image
    c *= ratio
    # Convert c to int
    c = c.astype("int")
    # Draw the contour
    cv2.drawContours(image , [c] , -1 , (0 , 255 , 0) , 2)
    # Put the name of the shape as text
    cv2.putText(image , shape , (cX , cY) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255 , 255 , 255) , 2)
    # Display the output image that has both the contours and the name of the shape
    cv2.imshow("Image", image)
    cv2.waitKey(0)

# Note:- Press Enter to detect the shapes one by one.