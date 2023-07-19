import gym
import rospy
import numpy as np
import time
import math
import random
from tf.transformations import quaternion_from_euler
from gym import spaces
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState

class turtlebot_env(gym.Env):
    def __init__(self):
        
        # Define the ROS node and subscriber
        
        rospy.Subscriber("/odom", Odometry, self._odom_callback)
        rospy.Subscriber("/imu", Imu, self._imu_callback)
        rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)
        self.odom = None
        self.imu = None
        self.scan = None
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Define the action space and observation space
        action_low = np.array([0.0, -0.5])  
        action_high = np.array([1.0, 0.5])
        self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)
        #self.observation_space = gym.spaces.Box(low=0, high=10, shape=(130,))
        self.reward_range = (-np.inf, np.inf)
        
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self._check_laser_scan_ready()
        num_laser_readings = len(laser_scan.ranges)
        self.new_ranges = 36
        # We only use two integers
        self.observation_space = spaces.Box(0, 7, shape=(self.new_ranges+4,))
         
        self.desired_point = Point()
        self.desired_point.x = random.uniform(2, 3.5)
        self.desired_point.y = random.uniform(-4.2, -2.2)
        # self.desired_point.x = random.uniform(-7.1, -6.1)
        # self.desired_point.y = random.uniform(-4.2, -2.2) 

        
        # Initialize the robot state and observation
        
        self.cumulated_steps = 0
        self.cumulated_reward = 0.0
        self._episode_done = False
    
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        initial_x = random.uniform(6, 7)
        initial_y = random.uniform(-4, -3)
        initial_z = 0.0
        self.desired_point.x = random.uniform(2, 3.5)
        self.desired_point.y = random.uniform(-4.2, -2.2)
        # self.desired_point.x = random.uniform(-7.1, -6.1)
        # self.desired_point.y = random.uniform(-4.2, -2.2)
        self._set_robot_state(initial_x, initial_y, initial_z, self.desired_point.x, self.desired_point.y)  

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        self.cumulated_steps = 0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        odometry = self._check_odom_ready()
        self.observation = self._get_obs()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose.pose.position)
        return True
    
    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        # action[0] = abs(action[0])
        linear_speed, angular_speed = action[0], action[1]
        """ if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "TURN_RIGHT" """
        
        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
        self.last_action = action
        rospy.logdebug("END Set Action ==>"+str(action))
    
    def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return: 
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("TurtleBot3 Base Twist Cmd>>" + str(cmd_vel_value))
        self.cmd_vel_pub.publish(cmd_vel_value)
        # time.sleep(0.1)

    def _get_obs(self, linear_speed=0.0, angular_speed=0.0):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self._check_laser_scan_ready()
        
        discretized_observations = self.discretize_scan_observation(laser_scan, self.new_ranges)
        # self._episode_done = self.is_crashed(laser_scan.ranges)
        # We get the odometry so that SumitXL knows where it is.
        odometry = self._check_odom_ready()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y
        orientation_q = self.odom.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        angle_turtlebot = yaw

        # We round to only two decimals to avoid very big Observation space
        odometry_array = [round(x_position, 2),round(y_position, 2), round(yaw, 2)]
        
        #Get the desired goal position
        x_goal = self.desired_point.x
        y_goal = self.desired_point.y
        
        current_distance_turtlebot_target = math.sqrt((x_goal - x_position)**2 + (y_goal - y_position)**2)
        angle_turtlebot_target = math.atan2(y_goal - y_position, x_goal - x_position)
        angle_diff = angle_turtlebot_target - angle_turtlebot

        if angle_diff < -math.pi:
            angle_diff = angle_diff + 2*math.pi
        if angle_diff > math.pi:
            angle_diff = angle_diff - 2*math.pi

        target_array = (round(current_distance_turtlebot_target, 2), round(angle_diff, 2))
        vel_array = (round(linear_speed, 2), round(angular_speed, 2))
        # We only want the X and Y position and the Yaw

        observations = tuple(discretized_observations) + target_array + vel_array 
        rospy.logdebug("Observations==>"+str(observations))
        rospy.logdebug("END Get Observation ==>")
        return np.array(observations)
    
    def _is_done(self, odometry, laser_scan):
        
        if self.is_crashed(laser_scan):
            self._episode_done = True
        elif self.is_in_desired_position(odometry.pose.pose.position):
            # rospy.logdebug("TurtleBot3 is NOT close to a wall ==>")
            # current_position = Point()
            # current_position.x = odometry.pose.pose.position.x
            # current_position.y = odometry.pose.pose.position.y
            # current_position.z = 0.0
            # # We see if it got to the desired point
            # if self.is_in_desired_position(current_position):
            self._episode_done = True
        return self._episode_done
    
    def _check_laser_scan_ready(self):
        rospy.logdebug("Waiting for /scan to be READY...")
        while self.scan is None and not rospy.is_shutdown():
            try:
                self.scan = rospy.wait_for_message("/scan", LaserScan, timeout=1.0)
                rospy.logdebug("Current /scan READY=>")

            except:
                rospy.logerr("Current /scan not ready yet, retrying for getting laser_scan")
        return self.scan

    def _check_odom_ready(self):
        rospy.logdebug("Waiting for /odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/odom", Odometry, timeout=1.0)
                rospy.logdebug("Current /odom READY=>")

            except:
                rospy.logerr("Current /scan not ready yet, retrying for getting laser_scan")
        return self.odom
    

    def _compute_reward(self, observations, odometry, action, done):
        current_position = odometry.pose.pose.position

        distance_from_des_point = self.get_distance_from_desired_point(current_position)

        distance_reward = self.previous_distance_from_des_point - distance_from_des_point
        # print("distance_from_des_point={}".format(distance_from_des_point))
        # print("previous_distance_from_des_point={}".format(self.previous_distance_from_des_point))
        # print("distance_reward={}".format(distance_reward))
        min_laser_range = min(observations[:-4])
        reward_collision = 0
        for i in range(len(observations[:-4])):
            if 0.2 < observations[i] < 0.4:
                reward_collision += -20
            elif min_laser_range < 0.2:
                reward_collision += -100
        
        angular_punish_reward = 0
        linear_punish_reward = 0

        if action[1] > 0.8:
            angular_punish_reward = -1
        if action[1] < -0.8:
            angular_punish_reward = -1

        if action[0] < 0.1:
            linear_punish_reward = -4
        
        arrive_reward = 0
        if done:
            if self.is_in_desired_position(current_position):
                print("got it!!!!!!!!!!!!!!!!!")
                arrive_reward = 2500
            else:
                arrive_reward = -2000
            
        reward = distance_reward*300 + reward_collision + angular_punish_reward + linear_punish_reward + arrive_reward
        
        # print("reward={}".format(reward))
        self.previous_distance_from_des_point = distance_from_des_point
        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
        return reward
    
    def reset(self):
        # Reset the robot to its initial state
        self._set_init_pose()
        self._init_env_variables()
        # Wait for the first robot state message to arrive
        self.observation = self._get_obs()
        rospy.sleep(0.2)
        return self.observation

    def step(self, action):
        # Apply the action to the robot
        self._set_action(action)
        # Wait for the next robot state message to arrive
        rospy.sleep(0.2)
        # Update the observation and compute the reward
        new_observation = self._get_obs(linear_speed=action[0], angular_speed=action[1])
        odometry = self._check_odom_ready()
        self._is_done(odometry, new_observation[:-4])
        done = self._episode_done
        reward = self._compute_reward(new_observation, odometry, action, done)
        # Set the new observation and done flag
        self.observation = new_observation
        
        return new_observation, reward, done, {}


    def _set_robot_state(self, rx, ry, rz, gx, gy):
        # Set the robot state using a ROS publish 
        yaw= 3.12
        qx, qy, qz, qw = quaternion_from_euler(0, 0, yaw)  # roll and pitch are 0
        pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        model_state = ModelState()
        model_state.model_name = 'turtlebot3_waffle'
        model_state.pose = Pose()
        model_state.pose.position.x = rx
        model_state.pose.position.y = ry
        model_state.pose.position.z = rz
        model_state.pose.orientation.x = qx
        model_state.pose.orientation.y = qy
        model_state.pose.orientation.z = qz
        model_state.pose.orientation.w = qw
        model_state.twist = Twist()
        model_state.reference_frame = 'world'
        rospy.sleep(1)  # wait for publisher to connect
        pub.publish(model_state)

        goal_state = ModelState()
        goal_state.model_name = 'beer'
        goal_state.pose = Pose()
        goal_state.pose.position.x = gx
        goal_state.pose.position.y = gy
        goal_state.pose.position.z = rz
        goal_state.twist = Twist()
        goal_state.reference_frame = 'world'
        rospy.sleep(1)  # wait for publisher to connect
        pub.publish(goal_state)

    def is_crashed(self, laser_scan):
        """
        It returns True if it has crashed based on the laser readings
        """
        min_range = 0.2
        done = False
        count = 0
        for i, item in enumerate(laser_scan):
            if min_range > item > 0:
                rospy.logdebug("TurtleBot is Too Close to wall==>")
                count += 1
                # print("laserscan={}".format(item))
        if count > 3:
            done = True
            #rospy.logerr("TurtleBot CRASHED==>")
            
        return done
    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        
        discretized_ranges = []
        mod = len(data.ranges)//new_ranges
        rospy.logdebug("data=" + str(data))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("mod=" + str(mod))
        
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or np.isinf(item):
                    discretized_ranges.append(6.21)
                elif np.isnan(item):
                    discretized_ranges.append(0.0)
                else:
                    discretized_ranges.append(item)

        return discretized_ranges
    
    def is_in_desired_position(self, current_position, epsilon=0.5):
        """
        It return True if the current position is similar to the desired poistion
        """
        
        is_in_desired_pos = False
        
        
        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon
        
        x_current = current_position.x
        y_current = current_position.y
        #print(x_current,y_current)
        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)
        
        is_in_desired_pos = x_pos_are_close and y_pos_are_close
        
        return is_in_desired_pos

    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)
        return distance
    
    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array((pstart.x, pstart.y, pstart.z))
        b = np.array((p_end.x, p_end.y, p_end.z))
    
        distance = np.linalg.norm(a - b)
    
        return distance
    
    def _odom_callback(self, data):
        self.odom = data
    
    def _imu_callback(self, data):
        self.imu = data

    def _laser_scan_callback(self, data):
        self.scan = data
    
    