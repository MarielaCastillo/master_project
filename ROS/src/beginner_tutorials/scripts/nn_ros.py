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




# Definition of dummy model
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = F.relu(x)
      x = F.max_pool2d(x, 2)
      x = self.dropout1(x)
      x = torch.flatten(x, 1)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)
      output = F.log_softmax(x, dim=1)
      return output

class CNNExpert1(nn.Module):
    def __init__(self, inchannel, numclasses):
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b4', in_channels=inchannel)
        self.disable_gradients(self.efficient_net)
        self.e2h = nn.Linear(1792, 64)
        self.hidden_dim = 64
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=256),
            #nn.Linear(in_features=1792, out_features=256),
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
        random_input_img = torch.rand((1, 1, 28, 28))    # random 28x28 image
        random_output = my_nn(random_input_img)
        rospy.loginfo('Here is the NN Output:')
        rospy.loginfo(random_output)
    '''


    if my_global_variable:
      w, h = 128, 128
      random_input_img = torch.rand(8, 3, w, h)
      random_output = my_nn(random_input_img)
      rospy.loginfo('Here is the NN Output:')
      rospy.loginfo(random_output)
    '''





# Node body
def main():    
    rospy.loginfo('Node: Main body')
    rospy.Subscriber("bool_topic", Bool, callback)

    # NN init
    global my_nn
    my_nn = Net()
    #my_nn = CNNExpert1(3,3)

    rospy.spin()



# Node initialization
if __name__ == "__main__":
    rospy.init_node('simple_pytorch_nn')
    rospy.loginfo('Node initialized')
    main()
