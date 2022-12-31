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

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os


def talker3():
    pub = rospy.Publisher('chatter3', Image, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        bridge = CvBridge()

        folder_path = '/home/miw/Documents/MasterProject/KITTI/blackandwhite/'
        images = []
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path,filename))
            if img is not None:
                images.append(img)

    
        try:
            for item in images:
                rospy.loginfo(item)
                msg = bridge.cv2_to_imgmsg(item, "bgr8")
                pub.publish(msg)
            rate.sleep() 

        except CvBridgeError as e:
            print(e) 

if __name__ == '__main__':
    try:
        talker3()
    except rospy.ROSInterruptException:
        pass



'''
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

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os


def talker():
    pub = rospy.Publisher('chatter3', Image, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        # hello_str = "hello world %s" % rospy.get_time()
        
        # rospy.loginfo(hello_str)
        # pub.publish(hello_str)
        
        #rate.sleep()
        
        # path
        
        #path = '/home/miw/Downloads/00/fl_ir_aligned/fl_ir_aligned_1570722156_952177040.png'
        path = '/home/miw/Downloads/00/fl_rgb/fl_rgb_1570722156_952177040.png'
        path = '/home/miw/Documents/MasterProject/KITTI/blackandwhite/0000000000.png'
        
        # Using cv2.imread() method
        img = cv2.imread(path, cv2.IMREAD_COLOR) ### para monocromaatico IMREAD_ no color

        # Create the cv_bridge object
        bridge = CvBridge()


        #folder_path = '/home/miw/Downloads/00/Test/'
        #folder_path = '/home/miw/Downloads/00/fl_rgb/'
        
        
        #folder_path = '/home/miw/Documents/MasterProject/KITTI/blackandwhite/'
        folder_path = '/home/miw/Documents/MasterProject/KITTI/rgb/'
        images = []
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path,filename))
            if img is not None:
                images.append(img) ### revisar kaljhdfjkh

    
        try:
            for item in images:
                rospy.loginfo(item)
                msg = bridge.cv2_to_imgmsg(item, "bgr8")
                pub.publish(msg)
            rate.sleep()  ###### todaviia no sirve kahsfkjahsdjflahdsfkh

        except CvBridgeError as e:
            print(e) 

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
'''