#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, "/home/iclab/.local/lib/python3.5/site-packages/")

from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import rospy
from yolo_V4.msg import ROI,ROI_array

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():
    pub = rospy.Publisher('chatter', ROI_array, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    global metaMain, netMain, altNames
    configPath = "./src/yolo_V4/src/yolov4.cfg"
    weightPath = "./src/yolo_V4/src/yolov4.weights"
    metaPath = "./src/yolo_V4/src/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    # out = cv2.VideoWriter(
    #     "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
    #     (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")
    
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while not rospy.is_shutdown():
        # prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        center_x=0.0
        center_y=0.0
        item=""
        #print(detections)
        
        ROI_array_all=ROI_array()
        
        for i in range(len(detections)):
            ROI_one=ROI()
            
            item=detections[i][0].decode()
            center_x=round(detections[i][2][0],3)
            center_y=round(detections[i][2][1],3)
            center_w=round(detections[i][2][2],3)
            center_h=round(detections[i][2][3],3)
            print("item : {}".format(item))
            
            ROI_one.object_name=item
            ROI_one.score=detections[i][1]
            ROI_one.min_x=int(center_x-center_w//2)
            ROI_one.Max_x=int(center_x+center_w//2)
            ROI_one.min_y=int(center_y-center_h//2)
            ROI_one.Max_y=int(center_y+center_h//2)

            ROI_array_all.ROI_list.append(ROI_one)
        
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)

        
        pub.publish(ROI_array_all)
        print("-----------------------------------")
        cv2.waitKey(3)
    cap.release()
    # out.release()

if __name__ == "__main__":
    try:
        YOLO()
    except rospy.ROSInterruptException:
        pass
