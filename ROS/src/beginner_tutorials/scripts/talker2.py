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
from std_msgs.msg import String
import os
import numpy as np
import pandas as pd

def read_text_file(file_path):
    print("file_path:", file_path)
    print(type(file_path))
    df = pd.read_csv(file_path, sep=" ", header=None)
    

    df = df.drop(df.columns[[3]], axis=1)
    df.columns.name = None
    df.reset_index(drop=True, inplace=True)
    return df

def read_text_file2(file_path): ##Correct but takes time
    #values_pointcloud = []
    values_pointcloud = np.zeros((1,3))
    with open(file_path, 'r') as f:
        print(file_path)
        for line in f: 
            values = line.split(' ')
            #values_pointcloud = np.r_[values_pointcloud,[[7,8,9]]]
            values_pointcloud = np.append(values_pointcloud, [[float(values[0]), float(values[1]), float(values[2])]], axis = 0)

    print("values", values)
    print("values_pointcloud", values_pointcloud)
    print(len(values_pointcloud))
    print(len(values_pointcloud[0]))
    print(len(values_pointcloud[1]))
    print(len(values_pointcloud[2])) 

    #file = f.read()
    #values = file.split(' ')
    #return float(values[0]), float(values[1]), float(values[2])
    return values_pointcloud

def read_text_file1(file_path): ##just first line
    with open(file_path, 'r') as f:
        print(file_path)
        file = f.read()
        values = file.split(' ')
        return float(values[0]), float(values[1]), float(values[2])

def talker(point_cloud):
    pub = rospy.Publisher('chatter2', String, queue_size=10)
    rospy.init_node('talker2', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    
    #hello_str = "hello world %s" % rospy.get_time()
    hello_str = "hello world %s" % point_cloud

    rospy.loginfo(hello_str)
    pub.publish(hello_str)
    rate.sleep()

if __name__ == '__main__':
    try:
        #while not rospy.is_shutdown():
            folder_path = '/home/miw/Documents/MasterProject/KITTI/pointcloud'


                # iterate through all file
            for file in sorted(os.listdir(folder_path)):
                print("asdf file", file)
            # Check whether file is in text format or not
                if file.endswith(".txt"):
                    print("asdf4")
                    file_path = f"{folder_path}/{file}"

                    point_cloud = read_text_file(file_path)

                    talker(point_cloud)
                    print()

    except rospy.ROSInterruptException:
        pass