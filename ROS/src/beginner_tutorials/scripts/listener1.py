#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def callback_img(imgmsg): #mensaje

    ### agregar preprocesamiento/1. redimensionar
    #para entrenar, reduccion de dimensiones/canales (de 3 a 1)
    #forma simple (r+g+b)/3


    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgmsg) #imagen
    rospy.loginfo(type(img))
    cv2.imshow("nombre", img)
    cv2.waitKey(1)

def listener1():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('img', Image, callback_img)

    rospy.spin()

if __name__ == '__main__':
    listener1()
