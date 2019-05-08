import cv2
import numpy as np
import copy
import math
import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time

model_path = 'static.model'
label_path = 'lb.pickle'
#path = 'C:/Users/gaura/Documents/Signlanguage/test'
img_counter = 1
#from appscript import app

# Environment:
# OS    : Mac OS EL Capitan
# python: 3.5
# opencv: 2.4.13

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [0, 255, 0], -1)
            return True, cnt
    return False, 0


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

count = 0
model = load_model(model_path)
lb = pickle.loads(open(label_path, "rb").read())
previouslabel="-1"

label_list = list()

while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        global img
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
#        cv2.imshow('mask', img)
        
        '''
        cv2.imwrite("frame%d.jpg" % count, img)
        count += 1
        '''
        

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        #cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow('ori', thresh)


        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        _,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)  
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 255, 0), 3)

            isFinishCal,cnt = calculateFingers(res,drawing)
            if triggerSwitch is True:
                if isFinishCal is True and cnt <= 2:
                    print (cnt)
                    #app('System Events').keystroke(' ')  # simulate pressing blank space
                    

#        cv2.imshow('output', drawing)
#        cv2.imshow('combo',drawing + img)
        
        combo = drawing + img
        
#        cv2.imwrite("test.jpg", drawing + img)
        
        testImg = drawing + img
        testImg = cv2.resize(testImg, (20, 24))
        testImg = testImg.astype("float") / 255.0
        testImg = img_to_array(testImg)
        testImg = np.expand_dims(testImg, axis=0)
        
        proba = model.predict(testImg)[0]
        idx = np.argmax(proba)
        if (proba[idx] * 100 >85):
            label = lb.classes_[idx]
            if (label==previouslabel):
                time.sleep(0.5)
            else:
                previouslabel=label
                label_list.append(label)
            #label = "{}: {:.2f}% ".format(label, proba[idx] * 100)
        else:
            label=""
                
        #print(label)
        str1 = ''.join(label_list)
        #cv2.putText(combo, str1, (20, 75),  cv2.FONT_HERSHEY_SIMPLEX,3, (0, 255, 0), 2)
       # cv2.namedWindow('combo',cv2.WINDOW_NORMAL)
       # cv2.resizeWindow('combo', 600,600)
        cv2.imshow("combo", combo)
        imgblack = np.zeros((500, 800))
        cv2.putText(imgblack, str1, (20, 75),  cv2.FONT_HERSHEY_SIMPLEX,3, (255, 255, 0), 2)     
        cv2.imshow("imgblack", imgblack)


    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')
    elif k==ord('c'):
        #if label_list.count()>0:
        if len(label_list)!=0:
                del label_list[-1]
    elif k == 32:
        print("space")
        label_list.append(" ")

#    elif k%256 == 32:
#        # SPACE pressed
#        if(img_counter<1001):
#            img_name = "Dataset_test_{}.jpg".format(img_counter)
#            image = cv2.imwrite(os.path.join(path,img_name),img)
#            print("{} written!".format(img_name))
#            img_counter += 1
        
        

camera.release()
cv2.destroyAllWindows()
