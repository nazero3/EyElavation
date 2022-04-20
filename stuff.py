import cv2
import pyautogui
import numpy as np

def calc_center(eye):
    x, y, w, h = eye
    return (x + (w // 2)), (y + (h // 2))


def calc_scale(f1, f2):
    return (f1 // f2)


def calc_pog(f, scale, coi, coe):
    return (f // 2) + (scale * (coe - coi))


def adjust(pog_x, pog_y):
    screen_width = pyautogui.size().width
    screen_height = pyautogui.size().height
    if pog_x < 0:
        pog_x = 0
    if pog_x > screen_width:
        pog_x = screen_width
    if pog_y < 0:
        pog_y = 0
    if pog_y > screen_height:
        pog_y = screen_height
    return pog_x, pog_y

def min_circle(circles):
    min_r = 1000000
    cx = None
    cy = None
    for x,y,r in circles:
        if r < min_r:
            min_r = r
            cx = x
            cy = y
    return cx, cy, min_r


def sobel(img):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    mag = np.hypot(grad_x, grad_y)
    mag = mag / mag.max() * 255
    mag = np.uint8(mag)
    return mag

def getleftmosteye(eyes):
    if len(eyes) != 2:
        return eyes
    leftmost = 9999999
    leftmostindex = -1
    for i in range(2):
        if eyes[i][0] < leftmost:
            leftmost = eyes[i][0]
            leftmostindex = i
    return [eyes[leftmostindex]]