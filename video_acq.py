#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from time import localtime
import rospy
from sensor_msgs.msg import NavSatFix

cam = cv2.VideoCapture(1)
cam.set(3,  1920)
cam.set(4, 1080)


pos = ""


def recupere_pos(msg):
    ROS_ERROR(msg.latitude)
    pos = str(msg.latitude) + str(msg.longitude)


rospy.init_node('enregistre_image_pos')
rate = rospy.Rate(1)

sub_pos = rospy.Subscriber("/fix", NavSatFix, recupere_pos, queue_size=1)


while not rospy.is_shutdown():
    _, img = cam.read() # captures image
    now = localtime()
    cv2.imwrite("prise_{0}-{1}-{2}-{3}h{4}min{5}s.png".format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec), img)
    rate.sleep()
