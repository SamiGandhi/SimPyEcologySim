from random import random
import cv2 as cv
import numpy as np


mog = cv.createBackgroundSubtractorMOG2()
knn = cv.createBackgroundSubtractorKNN()
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

def SOBEL_EDGE_DETECTOR(img):
    # Convert to graycsale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)
    sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    return sobelxy


def SOBEL_EDGE_DETECTOR_GRAY(img):
    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(img, (3, 3), 0)
    sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    return sobelxy


def CALCULATE_SAD_TO_GRAY_SCAL_FRAMES(current_frame,previous_frame):

    if current_frame is None:
        current_frame = previous_frame

    framedelta = cv.absdiff(current_frame, previous_frame)
    retval, bgs = cv.threshold(framedelta.copy(), 50, 255, cv.THRESH_BINARY)
    return bgs

def BLUR_2D_RANK(image, kernel_size_x, kernel_size_y):
    blured_image = cv.blur(image, (kernel_size_x, kernel_size_y))
    return blured_image

def SOOTHING_USING_MEDIAN_BLUR(frame):
    medianBlur = cv.medianBlur(np.int8(frame),1)
    return medianBlur

# Define the roi extraction methods
def ahcen_method(frame_n, frame_n_1, threshold=30):
    if frame_n is None:
        return None
    # SOBEL EDGE DETECTION TO EACH FRAME
    sob_frame_n = SOBEL_EDGE_DETECTOR(frame_n)
    sob_frame_n_1 = SOBEL_EDGE_DETECTOR(frame_n_1)
    # CALCULATE THE SUM OF ABSOLUTE DIFFERENCES (SAD)
    sad = CALCULATE_SAD_TO_GRAY_SCAL_FRAMES(sob_frame_n, sob_frame_n_1)
    # 2D_RANK_ORDER_BLUR USING A 9X9 KERNEL
    blured_frame = BLUR_2D_RANK(sad, 9, 9)
    # SMOOTHING FRAME USING FAST GLOBAL SMOOTHING USING THE SIMPLE SMOOTHER WITHOUT GPU
    smoothed = SOOTHING_USING_MEDIAN_BLUR(blured_frame)
    height, width = smoothed.shape
    all_pixels = height * width
    number_of_black_pix = np.sum(smoothed == 0)
    region_of_interest_percentage = 100 - ((number_of_black_pix * 100) / all_pixels)
    if region_of_interest_percentage >= threshold:
        result = frame_n.copy()
        result[smoothed == 0] = (0, 0, 0)
        #pts = cv.findNonZero(result)
        #x, y, w, h = cv.boundingRect(pts)
        #bird = result[y:y + h, x:x + w]
        #return bird
        cv.imshow('Result', result)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return result
    else:
        return None


def ameliorated_method_mog2(frame, minimum=4000):
    vid = cv.flip(frame, 1)
    bgs = mog.apply(vid)
    contours, _ = cv.findContours(bgs, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    for cnt in contours:
        if cv.contourArea(cnt) < minimum:
            return None
        (x, y, w, h) = cv.boundingRect(cnt)
        # cv2.drawContours(bgs,cnt,-1,255,3)
        bird = vid[y:y + h, x:x + w]
        return bird


def ameliorated_method_knn(frame, minimum=4000):
    vid = cv.flip(frame, 1)
    bgs = knn.apply(vid)
    contours, _ = cv.findContours(bgs, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    for cnt in contours:
        if cv.contourArea(cnt) < minimum:
            return None
        (x, y, w, h) = cv.boundingRect(cnt)
        # cv2.drawContours(bgs,cnt,-1,255,3)
        bird = vid[y:y + h, x:x + w]
        return bird


def no_region_of_interest(frame):
    return frame


# Define the feature extraction methods
def akaze(roi):
    akaze = cv.AKAZE_create()
    return akaze.detectAndCompute(roi, None)


def kaze(roi):
    kaze = cv.KAZE_create()
    return kaze.detectAndCompute(roi, None)


def orb(roi):
    orb = cv.ORB_create()
    return orb.detectAndCompute(roi, None)


def fast(roi):
    fast = cv.FastFeatureDetector_create()
    return fast.detectAndCompute(roi, None)


def sift(roi):
    sift = cv.SIFT_create()
    return sift.detectAndCompute(roi, None)


# Matching features used methods
def brute_force_matcher(desc1, desc2):
    bf = cv.BFMatcher()
    if desc1 is not None and desc2 is not None:
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    else: return None


# The ratio threshold is taken by default as in Lowe implementation
def flann_matcher(desc1, desc2, ratio_thresh=0.7):
    knn_matches = matcher.knnMatch(desc1, desc2, 2)
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return good_matches


def process_matches(matches):
    if len(matches) > 20 and matches is not None:
        print("Redundant object")
    else:
        print("no redundant object")
    return None


def data_lost(img, max_box_number):
    rows, cols, channels = img.shape
    num_boxes = random.randint(0, 10)

    # Loop over the number of black boxes
    for i in range(num_boxes):
        # Generate random width and height of the box
        width = random.randint(10, 100)
        height = random.randint(10, 100)

        # Generate random x and y starting position of the box
        x = random.randint(0, cols - width)
        y = random.randint(0, rows - height)

        # Create a black box with the random size and position
        img[y:y + height, x:x + width, :] = 0
        return img