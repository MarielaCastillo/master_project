#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def callback3(imgmsg):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgmsg)
    rospy.loginfo(type(img))
    cv2.imshow("nombre", img)
    cv2.waitKey(1)
   
def listener3():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('chatter3', Image, callback3)
  
    rospy.spin()

if __name__ == '__main__':
    listener3()
