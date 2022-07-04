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
from std_msgs.msg import Float64MultiArray
import os
import numpy as np
import pandas as pd

from beginner_tutorials.msg import MyArray


def read_text_file(file_path):
    df = pd.read_csv(file_path, sep=" ", header=None)
    df = df.drop(df.columns[[3]], axis=1)

    return df


if __name__ == '__main__':
    try:
            
            rospy.init_node('talker2', anonymous=True)
            pub = rospy.Publisher('chatter2', MyArray, queue_size=10)
            rate = rospy.Rate(10) # 10hz

        #while not rospy.is_shutdown():

            folder_path = '/home/miw/Documents/MasterProject/KITTI/pointcloud'
            
            #my_array_for_publishing = Float64MultiArray()
            my_array_for_publishing = MyArray()
            print("layout ", my_array_for_publishing.layout)
            

                # iterate through all file
            for file in sorted(os.listdir(folder_path)):
            # Check whether file is in text format or not
                if file.endswith(".txt"):
                    file_path = f"{folder_path}/{file}"
                    print()
                    print("asdf1")
                    point_cloud = read_text_file(file_path)

                    hello_str = "hello world %s" % point_cloud

                    pcArray = point_cloud.to_numpy()
                    #print(len(pcArray))
                    #print(pcArray.shape)
                    #my_array_for_publishing.layout.dim = [len(pcArray),3]
                    #print("asdfpcArray",type(pcArray))
                    
                    
                    
                    #my_msg = Float64MultiArray() 
                    #my_msg.data = point_cloud
                    print()
                    #print("asdf2")
                    #print()
                    #print("asdf3",type(my_msg))
                    #print("asdf4",type(my_msg.data))
                    print()
                    #print("asdf5")
                    #rospy.loginfo(my_msg.data)

                    #print("test", pcArray)

                    #my_array_for_publishing = Float64MultiArray(data=pcArray)

                    
                    my_array_for_publishing.data = pcArray
                    #print("my_array_for_publishing ", my_array_for_publishing)

                    #print("asdfpcArray",type(my_array_for_publishing))

                    #pub.publish(pcArray)




                    my_array_for_publishing.layout.dim = [1]
                    my_array_for_publishing.data = [24]
                    print("my_array_for_publishing ", my_array_for_publishing)

                    print("asdfpcArray",type(my_array_for_publishing))

                    pub.publish(my_array_for_publishing)
                    print()
                    print("asdf6")
                    rate.sleep()

                    ## cambiar a Float64MultiArray
                    
            print()

    except rospy.ROSInterruptException:
        pass