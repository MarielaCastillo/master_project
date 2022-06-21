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

## Simple talker demo that listens to std_msgs/Strings published 
## to the 'chatter' topic

import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray

import numpy as np



def callback_img(imgmsg): #mensaje

    ### agregar preprocesamiento/1. redimensionar
    #para entrenar, reduccion de dimensiones/canales (de 3 a 1)
    #forma simple (r+g+b)/3


    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgmsg) #imagen
    rospy.loginfo(type(img))
    cv2.imshow("nombre", img)
    cv2.waitKey(1)
    global a
    a = img
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)

def callback2(data):
    received_2D_array = data.data
    rospy.loginfo("And here, reconstructed Matrix: ")
    length = data.layout.dim[0].size
    height = data.layout.dim[1].size
    reconstructed_2D_array = np.asarray(received_2D_array).reshape((length,height))
    rospy.loginfo("yay!")

def callback3(imgmsg):

    ### agregar preprocesamiento/1. redimensionar
    
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgmsg) #imagen
    rospy.loginfo(type(img))
    cv2.imshow("nombre", img)
    cv2.waitKey(1)
    global c
    c = img
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)

def callback4(data):
    rospy.loginfo(rospy.get_caller_id() + 'asdf rewq4 %s', data.data)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('img', Image, callback_img)
    rospy.Subscriber('topic2D', Float64MultiArray, callback2)
    rospy.Subscriber('chatter3', Image, callback3)
    rospy.Subscriber('chatter4', String, callback4)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

def network():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    model = LitModelEfficientNet(512, transform)
    #trainer = Trainer(gpus=1, max_epochs=2)
    trainer = Trainer(accelerator="cpu",max_epochs=2)
    trainer.fit(model)



if __name__ == '__main__':
    listener()
    #network() ###entrenar red con kitti y etiquetas del archivo. y ya usar con estos 3 valores





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

## Simple talker demo that listens to std_msgs/Strings published 
## to the 'chatter' topic

import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2

def callback_img(imgmsg): #mensaje

    ### agregar preprocesamiento/1. redimensionar
    #para entrenar, reduccion de dimensiones/canales (de 3 a 1)
    #forma simple (r+g+b)/3


    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgmsg) #imagen
    rospy.loginfo(type(img))
    cv2.imshow("nombre", img)
    cv2.waitKey(1)
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)

def callback2(data):
    rospy.loginfo(rospy.get_caller_id() + 'asdf %s', data.data)

def callback3(imgmsg):

    ### agregar preprocesamiento/1. redimensionar
    
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgmsg) #imagen
    rospy.loginfo(type(img))
    cv2.imshow("nombre", img)
    cv2.waitKey(1)
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)

def callback4(data):
    rospy.loginfo(rospy.get_caller_id() + 'asdf rewq4 %s', data.data)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('img', Image, callback_img)
    rospy.Subscriber('chatter2', String, callback2)
    rospy.Subscriber('chattedatar3', Image, callback3)
    rospy.Subscriber('chatter4', String, callback4)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()

'''