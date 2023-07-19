import rospy
from sensor_msgs.msg import LaserScan

class LaserScanner:
    def __init__(self):
        self.scan = None
        self.subscriber = rospy.Subscriber("/scan", LaserScan, self._scan_callback)

    def _scan_callback(self, data):
        self.scan = data
        rospy.loginfo("Scan ranges: %s", len(self.scan.ranges))

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('laser_scanner_node', anonymous=True)
        scanner = LaserScanner()
        scanner.run()
    except rospy.ROSInterruptException:
        pass
