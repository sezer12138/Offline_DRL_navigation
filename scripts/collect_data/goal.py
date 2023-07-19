import rospy
from geometry_msgs.msg import PoseStamped
import random

def set_goal(desired_x, desired_y):
    # # Initialize the node
    # rospy.init_node('random_goal_node')

    # Create a publisher to the /move_base_simple/goal topic
    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

    # Wait until connected
    rospy.sleep(1.0)

    # Generate a random goal within specific range
    goal = PoseStamped()
    goal.header.frame_id = "map"  # Use the map frame to define goal poses
    # goal.pose.position.x = random.uniform(-7, 7)  # Set your own range here
    # goal.pose.position.y = random.uniform(-4, -2.5)  # Set your own range here
    goal.pose.position.x = desired_x
    goal.pose.position.y = desired_y

    # Quaternion representation of yaw, pitch, roll
    goal.pose.orientation.z = random.uniform(-1, 1)  # random orientation (range -1 to 1)
    goal.pose.orientation.w = 1.0  # we assume upright (non-changing) orientation

    # Publish the goal
    pub.publish(goal)

if __name__ == "__main__":
    set_goal()
