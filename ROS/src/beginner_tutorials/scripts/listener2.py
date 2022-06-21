#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64MultiArray
from beginner_tutorials.msg import MyArray


from std_msgs.msg import String

def callback2(data):
    #rospy.loginfo(rospy.get_caller_id() + 'asdf %s', data.data)
    print("hi4")
    rospy.loginfo(type(data))
    rospy.loginfo(data)


def listener2():
    print("hi1")
    rospy.init_node('listener', anonymous=True)
    print("hi2")
    rospy.Subscriber('chatter2', MyArray, callback2)
    print("hi3")

    rospy.spin()

if __name__ == '__main__':
    listener2()