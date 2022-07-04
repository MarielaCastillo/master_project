#!/usr/bin/env python

# Example on how to fill out data for Float64MultiArray msg
# have to be sent as long flat vectors first and then decompose in the listener


import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayDimension
import numpy as np

import pandas as pd
import os

def read_text_file(file_path):
    df = pd.read_csv(file_path, sep=" ", header=None)
    df = df.drop(df.columns[[3]], axis=1)

    return df

if __name__ == '__main__':
    try:    
        pub2D = rospy.Publisher('topic2D', Float64MultiArray, queue_size=10)

        rospy.init_node('talkerForArray', anonymous=True)
        rate = rospy.Rate(10) 

        msg_array_2D = Float64MultiArray()
        msg_array_2D2 = Float64MultiArray()

        
        folder_path = '/home/miw/Documents/MasterProject/KITTI/pointcloud'
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".txt"):
                    file_path = f"{folder_path}/{file}"
                    point_cloud = read_text_file(file_path)
                    length = len(point_cloud)
                    height = 3
                    msg_array_2D2.layout.dim = [ MultiArrayDimension() , MultiArrayDimension()]
                    msg_array_2D2.layout.dim[0].label = "This is higher order dimension"  
                    msg_array_2D2.layout.dim[0].size = length                        # define per each dimension
                    msg_array_2D2.layout.dim[0].stride = height ### check, why?
                    
                    msg_array_2D2.layout.dim[1].label = "This is lower order dimension"  
                    msg_array_2D2.layout.dim[1].size = height                        # define per each dimension
                    msg_array_2D2.layout.dim[1].stride = height ### check, why?

                    # before publishing, submit data first as a flat array of 24 elements
                    msg_array_2D2.data = point_cloud.to_numpy().flatten()

                    pub2D.publish(msg_array_2D2)
            
        rate.sleep()

                
    except rospy.ROSInterruptException: pass