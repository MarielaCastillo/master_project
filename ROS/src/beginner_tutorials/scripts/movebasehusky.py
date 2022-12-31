#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

from numpy import array
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import sys

def mover():
    pub = rospy.Publisher('husky_velocity_controller/cmd_vel', Twist, queue_size=1000) #move_base #rostopic list
    rospy.init_node('mover', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    vel = Twist()
    i = 0
    xi = 100000
    angz = 1

    
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        time = rospy.get_time()
        rospy.loginfo(hello_str)
        
        if i % 36 == 0:
            angz = angz * (-1)


        vel.linear.x = xi #(time-1652363160)/10 #revisar formmula
        vel.linear.y = 0
        vel.linear.z = 0
        vel.angular.x = 0
        vel.angular.y = 0
        vel.angular.z = angz   #(time-1652363160)/100 #revisar formmula #1
        #only change angular.z
        #only change linear.x
        i = i + 1
        
        rospy.loginfo("x = %f", vel.linear.x)
        rospy.loginfo("z = %f", vel.angular.z)

        rospy.loginfo("i = %f", i)
        rospy.loginfo("angz = %f", angz)

        pub.publish(vel)
        rate.sleep()

if __name__ == '__main__':
    try:
        mover()
    except rospy.ROSInterruptException:
        pass
