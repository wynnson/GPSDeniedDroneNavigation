import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from omegaconf.dictconfig import DictConfig

from src.database.tile_db_manager import TileDatabaseManager
from src.inference.localizer import Localizer
from src.utils.config import load_config


class VisualLocalizationNode(Node):
    def __init__(self, config: DictConfig):
        super().__init__('visual_localization_node')
        self.bridge = CvBridge()

        self.db_manager = TileDatabaseManager(
            db_path=config.output.db,
            faiss_path=config.output.faiss,
            embedding_dim=config.model.embedding_dim,
        )

        self.localizer = Localizer(
            config,
            self.db_manager,
        )

        self.subscription = self.create_subscription(
            Image,                      # message type
            "/camera/image",            # listen to this
            self.image_callback,        # call this as Image arrives
            10,                         # queue depth
        )

        self.publisher = self.create_publisher(
            PointStamped,
            "/visual_position",
            10
        )

    def image_callback(self, msg: Image):
        """Called when message arrvies in the queue"""
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        predictions = self.localizer.predict(image)
        lon, lat = self.localizer.estimate_position(predictions)

        output = PointStamped()
        output.header = msg.header
        output.point.x = float(lon)
        output.point.y = float(lat)

        self.get_logger().info(f"{lon}, {lat}")

        # publish an estimated location downstream
        self.publisher.publish(output)

    def destroy_node(self):
        """Cleanup node"""
        self.db_manager.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
        
    config = load_config("src/config/default_onnx.yaml")

    node = VisualLocalizationNode(config)

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
