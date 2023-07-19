import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Empty
import torch
import signal
import numpy
import sys
from env import turtlebot_env
from goal import set_goal

# initialize ROS node
rospy.init_node('teleop_to_data')

env = turtlebot_env()

class VelocityGetter:
    def __init__(self):
        self.velocity = Twist()
        rospy.Subscriber('/cmd_vel', Twist, self.callback)

    def callback(self, data):
        self.velocity = data

    def get_velocity(self):
        linear_velocity = self.velocity.linear.x
        angular_velocity = self.velocity.angular.z
        return linear_velocity, angular_velocity

velocity_getter = VelocityGetter()

# def get_keyboard_action():
#     linear_velocity, angular_velocity = velocity_getter.get_velocity()
#     action = [linear_velocity, angular_velocity]
#     return action

def signal_handler(sig, frame):
    # load data from file
    existing_state_tensor, existing_action_tensor, existing_reward_tensor, existing_new_state_tensor, existing_done_tensor = torch.load('new_data1.pth')

    # convert new data into tensor
    state_tensor = torch.tensor(numpy.array([t[0] for t in data]))
    action_tensor = torch.tensor(numpy.array([t[1] for t in data]))
    reward_tensor = torch.tensor(numpy.array([t[2] for t in data]))
    new_state_tensor = torch.tensor(numpy.array([t[3] for t in data]))
    done_tensor = torch.tensor(numpy.array([t[4] for t in data]))

    # concatenate existing and new data
    state_tensor = torch.cat((existing_state_tensor, state_tensor))
    action_tensor = torch.cat((existing_action_tensor, action_tensor))
    reward_tensor = torch.cat((existing_reward_tensor, reward_tensor))
    new_state_tensor = torch.cat((existing_new_state_tensor, new_state_tensor))
    done_tensor = torch.cat((existing_done_tensor, done_tensor))

    # save data using PyTorch
    torch.save((state_tensor, action_tensor, reward_tensor, new_state_tensor, done_tensor), 'new_data1.pth')
    sys.exit(0)

# handle Ctrl+C
signal.signal(signal.SIGINT, signal_handler)


# store data
data = []

# get initial state
state = env.reset()
# set_goal
set_goal(env.desired_point.x, env.desired_point.y)


while not rospy.is_shutdown():
    # get action from keyboard
    action = velocity_getter.get_velocity()

    rospy.sleep(0.5)
    # get new state after action
    new_state = env._get_obs(linear_speed=action[0], angular_speed=action[1])

    odom = env._check_odom_ready()

    done = env._is_done(odom, state[:-4])

    # calculate reward
    reward, _, _, _, _ = env._compute_reward(state, odom, action, done)


    # store state, action, reward, new_state, done
    print("action:", action)
    print("reward:", reward)
    print("done:", done)
    data.append((state, action, reward, new_state, done))

    # if episode is done, reset state
    if done:
        state = env.reset()
        # set_goal
        set_goal(env.desired_point.x, env.desired_point.y)
    else:
        # update state
        state = new_state
