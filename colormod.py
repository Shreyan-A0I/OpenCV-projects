import cv2
import numpy as np

# Function to update lower HSV range
def update_lower_h(val):
    global lower_blue
    lower_blue[0] = val

def update_lower_s(val):
    global lower_blue
    lower_blue[1] = val

def update_lower_v(val):
    global lower_blue
    lower_blue[2] = val

# Function to update upper HSV range
def update_upper_h(val):
    global upper_blue
    upper_blue[0] = val

def update_upper_s(val):
    global upper_blue
    upper_blue[1] = val

def update_upper_v(val):
    global upper_blue
    upper_blue[2] = val

cap = cv2.VideoCapture(0)

cv2.namedWindow('Result')
cv2.namedWindow('Mask')

# Initialize HSV range for blue color
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Create trackbars for lower HSV range
cv2.createTrackbar('Lower H', 'Result', lower_blue[0], 180, update_lower_h)
cv2.createTrackbar('Lower S', 'Result', lower_blue[1], 255, update_lower_s)
cv2.createTrackbar('Lower V', 'Result', lower_blue[2], 255, update_lower_v)

# Create trackbars for upper HSV range
cv2.createTrackbar('Upper H', 'Result', upper_blue[0], 180, update_upper_h)
cv2.createTrackbar('Upper S', 'Result', upper_blue[1], 255, update_upper_s)
cv2.createTrackbar('Upper V', 'Result', upper_blue[2], 255, update_upper_v)

snapshot_count = 0

while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)
    
    # Show the original, mask, and result images
    cv2.imshow('Capturing...', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', res)
    
    # Capture snapshots on 's' key press
    k = cv2.waitKey(10)
    if k == ord('s'):
        cv2.imwrite(f'original_snapshot_{snapshot_count}.png', img)
        cv2.imwrite(f'processed_snapshot_{snapshot_count}.png', res)
        cv2.imwrite(f'mask_snapshot_{snapshot_count}.png', mask)
        snapshot_count += 1
    
    # Break the loop on 'Esc' key press
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
