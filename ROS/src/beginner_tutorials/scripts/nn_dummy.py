#!/usr/bin/python3

# Simple ROS node for python3 with pytorch
# When a TRUE (boolean msg) is received on topic "bool_topic",
# a randomly created 28x28 image is fed into a (non trained) dummy NN

# Requirements:
# - conda environment with pytorch, python3, python3-catkin-tools and rospkg
#   - sudo apt-get install python3-catkin-tools .. etc
# - ROS melodic

# To run:
# - conda activate
# - when compiling the ROS pkg: 
#         export ROS_PYTHON_VERSION=3
#         catkin config --cmake-args -DPYTHON_VERSION=3
#         catkin build 
# - source devel/setup.bash
# - rosrun my_pkg my_script.py  
#         if it doesnt work like this, then: python src/pytorch_nn_test/src/nn_dummy.py 
# 
# - From another terminal, publish TRUE or FALSE to topic "bool_topic" (from a publisher node or from cli)
#         $ rostopic pub /bool_topic std_msgs/Bool "data: true"




import rospy, logging
from std_msgs.msg import Bool

# For pytorch NN
import torch
import torch.nn as nn
from torch.nn import functional as F

from efficientnet_pytorch import EfficientNet
from collections import OrderedDict


class CNNExpert1(nn.Module):
    def __init__(self, inchannel, numclasses):
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b4', in_channels=inchannel)
        self.disable_gradients(self.efficient_net)
        self.e2h = nn.Linear(1792, 64)
        self.hidden_dim = 64
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=256),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=numclasses)
        )

    def disable_gradients(self, model):
        # update the pretrained model
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, input): ### 8, 3, 128, 128
        features = self.efficient_net.extract_features(input)
        hidden_state = self.efficient_net._avg_pooling(features)
        hidden_state = hidden_state.flatten(start_dim=1)
        hidden_state = self.e2h(hidden_state)
        return hidden_state, self.fc(hidden_state) #8x2 1792x28 error






# Callback to topic listening
def callback(data):
    global my_global_variable
    rospy.loginfo("On callback: %s", data.data)
    my_global_variable = data.data


    #This can be proposed as you wish.. data process of received data into the NN
    if my_global_variable:
      batch_size = 8
      w, h = 128, 128

      labels = torch.rand(batch_size, 3)
      inputrgb = torch.rand(batch_size, 3, w, h)
      inputdepth = torch.rand(batch_size, 1, w, h)
      inputLidar = torch.rand(batch_size, 2, w, h)
      inputThermo = torch.rand(batch_size, 1, w, h)
      

      hidden_statergb, outrgb = cnnexpertRGB(inputrgb)
      hidden_statedepth, outdepth = cnnexpertDepth(inputdepth)
      hidden_stateLidar, outLidar = cnnexpertLidar(inputLidar)
      hidden_stateThermo, outThermo = cnnexpertThermo(inputThermo)

      outconcatgating = torch.cat([hidden_statergb, hidden_statedepth, hidden_stateLidar, hidden_stateThermo], dim = -1)
      outconcatexpertclassifier = torch.stack([outrgb, outdepth, outLidar, outThermo], dim = -1)
      gating = gatingnetwork(outconcatgating)
      gating = gating.unsqueeze(1)
      outfinal = outconcatexpertclassifier * gating
      outfinal = outfinal.sum(-1)

      #outputs = self(inputrgb, inputdepth, inputLidar, inputThermo) #forward(x1, x2, x3, x4)
      #loss = self.criterion(outputs, labels)



      #random_output = my_nn(random_input_img)




      rospy.loginfo('Here is the NN Output:')
      rospy.loginfo(outfinal)
      #rospy.loginfo(outrgb)
    





# Node body
def main():    
    rospy.loginfo('Node: Main body')
    rospy.Subscriber("bool_topic", Bool, callback)

    # NN init
    #global my_nn
    global cnnexpertRGB, cnnexpertDepth, cnnexpertLidar, cnnexpertThermo
    global gatingnetwork
    #my_nn = Net()
    #my_nn = CNNExpert1(3,3)


    cnnexpertRGB = CNNExpert1(3, 3)
    cnnexpertDepth = CNNExpert1(1, 3) 
    cnnexpertLidar = CNNExpert1(2, 3)
    cnnexpertThermo = CNNExpert1(1, 3)
    hidden_concat_dim = cnnexpertRGB.hidden_dim + cnnexpertDepth.hidden_dim + cnnexpertLidar.hidden_dim + cnnexpertThermo.hidden_dim
    gatingnetwork = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_concat_dim, 64)), #output 64
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(64, 4)),   #output 2
            ('softmax', nn.Softmax(dim=1))
        ]))

    
    rospy.spin()



# Node initialization
if __name__ == "__main__":
    rospy.init_node('simple_pytorch_nn')
    rospy.loginfo('Node initialized')
    main()
