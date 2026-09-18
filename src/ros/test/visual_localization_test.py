import cv2
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TestVisualLocalizationNode(Node):
    def __init__(self):
        super().__init__("test_visual_localization_node")

        self.publisher = self.create_publisher(
            Image,
            "/camera/image",
            10,
        )

        self.bridge = CvBridge()

        self.image = cv2.imread("data/query4.png")

        self.timer = self.create_timer(
            1.0,              # every 1 second
            self.publish_image
        )

    def publish_image(self):
        msg = self.bridge.cv2_to_imgmsg(
            self.image,
            encoding="bgr8",
        )

        self.publisher.publish(msg)

        self.get_logger().info("Published test image")


def main(args=None):
    rclpy.init(args=args)

    node = TestVisualLocalizationNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
