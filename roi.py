import numpy as np
import shutil
import cv2
import os

from skimage import io


def get_eyes(detector, image):
    """ 
    Detects eye contours on the image

    Arguments:
    detector -- detector model which has landmark prediction method 'get_landmarks'
    image -- numpy array of shape [h, w, 3], the target image where eyes needs to be extracted from

    Return:
    Tuple of two contours of shape (-1, 1, 2)
    """

    eye1, eye2 = detector.get_eyes(image)

    eye1 = np.array(eye1).reshape((-1, 1, 2)).astype(np.int32)
    eye2 = np.array(eye2).reshape((-1, 1, 2)).astype(np.int32)

    return eye1, eye2


def get_roi(countours, img_height, img_width):
    """
    Computes bounding box which must contain all the passed contours 
    
    Arguments:
    contours -- numpy array of shape [num_points, 1, 2]
    img_height -- int, image height
    img_width -- int, image width

    Returns:
    4 numbers representing x and y coordinates of bottom left and upper right corners of the resulting ROI
    """

    miny, maxy = float('inf'), -float('inf')
    minx, maxx = float('inf'), -float('inf')
    for item in countours[:, :]:
        x, y = item[0]
        miny = min(y, miny)
        maxy = max(y, maxy)
        
        minx = min(x, minx)
        maxx = max(x, maxx)
    
    offset = 15
    minx = max(minx - offset, 0)
    maxx = min(maxx + offset, img_width)
    miny = max(miny - offset, 0)
    maxy = min(maxy + offset, img_height)

    return int(minx), int(maxx), int(miny), int(maxy)


def extract_eye_regions(landmark_detector, image):
    """
    Calculates bounding box coordinates for eye region given image path,
    uses some landmark detector

    Arguments:
    landmark_detector -- model which is capable of finding facial landmarks on the image
    image -- image to extract landmarks from

    Returns:
    rois_coords -- list of tuples (x1, x2, y1, y2)
    """

    rows, cols, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    eye1, eye2 = get_eyes(landmark_detector, image)
    eyes = [eye1, eye2]

    rois_coords = [get_roi(eye, rows, cols) for eye in eyes]

    return rois_coords, eyes
