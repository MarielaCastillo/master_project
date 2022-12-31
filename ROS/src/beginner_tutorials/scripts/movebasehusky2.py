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

##BEFORE

from numpy import array
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import sys

def mover():
    pub = rospy.Publisher('husky_velocity_controller/cmd_vel', Twist, queue_size=1000) #move_base #rostopic list
    rospy.init_node('mover', anonymous=True)
    rate = rospy.Rate(1) # 10hz

    twist_list = []
    twist = Twist()
    twist.linear.x= 13500000
    twist.linear.y = 0
    twist.angular.z = 1
    twist_list.append(twist)
    '''
    twist2 = Twist()
    twist2.linear.x= 100000
    twist2.linear.y = 0
    twist2.angular.z = 0.5
    twist_list.append(twist2)

    twist3 = Twist()
    twist3.linear.x= 300000
    twist3.linear.y = 0
    twist3.angular.z = 1
    twist_list.append(twist3)
    
    twist4 = Twist()
    twist4.linear.x= 1003000
    twist4.linear.y = 0
    twist4.angular.z = .2
    twist_list.append(twist4)

  
    twist5 = Twist()
    twist5.linear.x= 12300231
    twist5.linear.y =0
    twist5.angular.z = 1
    twist_list.append(twist5)
    '''
  


    while not rospy.is_shutdown():
        for item in twist_list:
            rospy.loginfo(item)
            pub.publish(item)
    rate.sleep() #cambiar a delay, para que se manden a diferentes tiempos



        #rate.sleep()

if __name__ == '__main__':
    try:
        mover()
    except rospy.ROSInterruptException:
        pass
