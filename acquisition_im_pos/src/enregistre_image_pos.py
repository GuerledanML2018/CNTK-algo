#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from time import localtime
import rospy
from sensor_msgs.msg import NavSatFix
from PIL import Image
import piexif

cam = cv2.VideoCapture(0)
cam.set(3,  1920)
cam.set(4, 1080)


pos = ""
pos_meta = 0,0

def recupere_pos(msg):
    global pos, pos_meta
    pos = "long: " + str(msg.latitude) + " lat: " + str(msg.longitude)
    pos_meta = msg.latitude, msg.longitude


rospy.init_node('enregistre_image_pos')
rate = rospy.Rate(0.2)

sub_pos = rospy.Subscriber("/fix", NavSatFix, recupere_pos, queue_size=1)

def formateGPS(a, dtype):
    if a < 0:
        if dtype == 'lon':
            direction = 'W'
        else:
            direction = 'S'
    else:
        if dtype == 'lon':
            direction = 'E'
        else:
            direction = 'N'

    a *= 1e7

    return int(abs(a)), direction


while not rospy.is_shutdown():
    _, img = cam.read() # capture image
    now = localtime()
    temps = "{0}-{1}-{2}-{3}h{4}min{5}s".format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    
    if img is not None:
        
        if len(pos) > 1:
            f = open("/home/tough/Desktop/Images/pos.gps", 'a')
            f.write(temps + " " + pos + "\n")
            f.close()


            lat, lat_dir = formateGPS(pos_meta[0], 'lat')
            lon, lon_dir = formateGPS(pos_meta[1], 'lon')
            
            gps_ifd = {piexif.GPSIFD.GPSLatitude: [lat, 10000000],
            piexif.GPSIFD.GPSLatitudeRef: lat_dir,
            piexif.GPSIFD.GPSLongitude: [lon, 10000000],
            piexif.GPSIFD.GPSLongitudeRef: lon_dir,
            }

            exif_dict = {"GPS":gps_ifd}
            exif_bytes = piexif.dump(exif_dict)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(imgRGB, 'RGB')
            im.save("/home/tough/Desktop/Images/prise_" + temps + ".jpg", exif=exif_bytes)

        else:
            print("no signal")
            cv2.imwrite("/home/tough/Desktop/Images/prise_" + temps + ".jpg", img)
    else:
        print("image is None")
    print "hey!"
    rate.sleep()
