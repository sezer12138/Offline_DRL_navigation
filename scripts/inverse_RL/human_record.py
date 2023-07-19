import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Empty
import pickle
from env_turtlebot import turtlebot_env

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

def get_keyboard_action():
    linear_velocity, angular_velocity = velocity_getter.get_velocity()
    action = [linear_velocity, angular_velocity]
    return action

# store data
data = []

# get initial state
state = env.reset()

while not rospy.is_shutdown():
    # get action from keyboard
    action = get_keyboard_action()

    # get new state after action
    new_state = env._get_obs(linear_speed=action[0], angular_speed=action[1])

    odom = env._check_odom_ready()

    done = env._is_done(odom, state[:-4])

    # calculate reward
    reward = env._compute_reward(state, odom, action, done)


    # store state, action, reward, new_state, done
    print(state, action, reward, new_state, done)
    data.append((state, action, reward, new_state, done))

    # if episode is done, reset state
    if done:
        state = env.reset()
    else:
        # update state
        state = new_state

# save data using pickle
with open('human_data.pkl', 'wb') as f:
    pickle.dump(data, f)
