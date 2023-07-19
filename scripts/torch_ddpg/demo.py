import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion

class RobotOdom:
    def __init__(self):
        self.odom = None
        # Subscribing to the "/odom" topic
        self.subscriber = rospy.Subscriber("/odom", Odometry, self._odom_callback)

    def _odom_callback(self, data):
        self.odom = data
        
    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            if self.odom is not None:
                position = self.odom.pose.pose.position
                orientation_q = self.odom.pose.pose.orientation
                orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                _, _, yaw = euler_from_quaternion(orientation_list)
                rospy.loginfo("Position: %s, Yaw: %s", position, yaw)
            rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('odom_node', anonymous=True)
        odom = RobotOdom()
        odom.run()
    except rospy.ROSInterruptException:
        pass
