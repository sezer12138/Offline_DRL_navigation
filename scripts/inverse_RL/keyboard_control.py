import rospy
from geometry_msgs.msg import Twist
from pynput import keyboard

# Define velocities
linear_velocity = 0.0
angular_velocity = 0.0

# Define increments
linear_increment = 0.1
angular_increment = 0.1

# Define keys for control
forward_key = keyboard.Key.up
backward_key = keyboard.Key.down
left_key = keyboard.Key.left
right_key = keyboard.Key.right
shift_key = keyboard.Key.shift

# Define the publisher
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

# Define the callback for keyboard inputs
def on_press(key):
    global linear_velocity, angular_velocity
    if key == forward_key:
        linear_velocity += linear_increment
    elif key == backward_key:
        linear_velocity -= linear_increment
    elif key == left_key:
        angular_velocity += angular_increment
    elif key == right_key:
        angular_velocity -= angular_increment
    elif key == shift_key:  # Stop the turtlebot when space bar is pressed
        linear_velocity = 0.0
        angular_velocity = 0.0

# Define the callback for keyboard release
def on_release(key):
    pass
    # global linear_velocity, angular_velocity
    # if key in [forward_key, backward_key]:
    #     linear_velocity = 0.0
    # elif key in [left_key, right_key]:
    #     angular_velocity = 0.0

# Register the callbacks
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Initialize the node
rospy.init_node('keyboard_teleop')

# Set the rate
rate = rospy.Rate(10)  # 10 Hz

# Loop until shutdown
while not rospy.is_shutdown():
    # Create the twist message
    twist = Twist()
    twist.linear.x = linear_velocity
    twist.angular.z = angular_velocity

    # Publish the message
    pub.publish(twist)

    # Sleep to maintain the rate
    rate.sleep()

# from pynput import keyboard
# import rospy
# from geometry_msgs.msg import Twist

# class KeyboardController:
#     def __init__(self):
#         self.linear_velocity = 0
#         self.angular_velocity = 0
#         self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

#     def on_press(self, key):
#         try:
#             if key == keyboard.Key.up:
#                 self.linear_velocity += 0.1
#             elif key == keyboard.Key.down:
#                 self.linear_velocity -= 0.1
#             elif key == keyboard.Key.right:
#                 self.angular_velocity -= 0.1
#             elif key == keyboard.Key.left:
#                 self.angular_velocity += 0.1
#             elif key == keyboard.Key.space:  # Stop the turtlebot when space bar is pressed
#                 self.linear_velocity = 0
#                 self.angular_velocity = 0
#         except AttributeError:
#             pass

#     def on_release(self, key):
#         if key == keyboard.Key.esc:
#             # Stop listener
#             return False

#     def start(self):
#         # ... initialize rospy if not initialized ...
#         with keyboard.Listener(
#                 on_press=self.on_press,
#                 on_release=self.on_release) as listener:
#             listener.join()

#         rate = rospy.Rate(10) # 10hz
#         while not rospy.is_shutdown():
#             msg = Twist()
#             msg.linear.x = self.linear_velocity
#             msg.angular.z = self.angular_velocity
#             self.pub.publish(msg)
#             rate.sleep()

# if __name__ == "__main__":
#     rospy.init_node('keyboard_teleop')
#     kc = KeyboardController()
#     kc.start()
