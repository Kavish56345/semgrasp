"""
ROS 2 node for NLP intent extraction.

Subscribes to /user_command (std_msgs/String),
extracts intent via Ollama, and publishes validated JSON
to /task_manifest (std_msgs/String).
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from nlp_intent.ollama_client import extract_intent


class IntentExtractorNode(Node):
    """Extracts structured intents from natural language commands."""

    def __init__(self):
        super().__init__('intent_extractor')

        # Parameters
        self.declare_parameter('model', 'phi3')
        self.declare_parameter('ollama_url', 'http://localhost:11434/api/generate')
        self.declare_parameter('timeout', 60)

        self._model = self.get_parameter('model').value
        self._url = self.get_parameter('ollama_url').value
        self._timeout = self.get_parameter('timeout').value

        # Subscriber
        self.sub = self.create_subscription(
            String,
            '/user_command',
            self._on_command,
            10,
        )

        # Publisher
        self.pub = self.create_publisher(String, '/task_manifest', 10)

        self.get_logger().info(
            f'Intent extractor ready  [model={self._model}, timeout={self._timeout}s]'
        )

    def _on_command(self, msg: String):
        command = msg.data
        self.get_logger().info(f'Received command: "{command}"')

        intent = extract_intent(
            command,
            model=self._model,
            api_url=self._url,
            timeout=self._timeout,
        )

        out = String()
        out.data = json.dumps(intent)
        self.pub.publish(out)

        self.get_logger().info(f'Published intent: {out.data}')


def main(args=None):
    rclpy.init(args=args)
    node = IntentExtractorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
