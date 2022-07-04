#!/usr/bin/env python

# Example on how to recover data from Float64MultiArray msg
# have to be size recovered as they were first sent as a flat vector

import rospy
from std_msgs.msg import Float64MultiArray
import numpy as np

def callback(data):
    #Recover data, recreate 2Dimensions Matrix of size: length x height x width
    received_2D_array = data.data
    rospy.loginfo("This is the 2D vector: ")
    #rospy.loginfo(received_2D_array)

    rospy.loginfo("And here, reconstructed Matrix: ")
    length = data.layout.dim[0].size
    height = data.layout.dim[1].size
    reconstructed_2D_array = np.asarray(received_2D_array).reshape((length,height))
    #rospy.loginfo(reconstructed_2D_array)
    rospy.loginfo("yay!")

def subscriber():
    rospy.Subscriber('topic2D', Float64MultiArray, callback)

    rospy.init_node('subscriberToArray', anonymous=True)
    rospy.spin()

if __name__ == '__main__':
    subscriber()