"""
Turtlesim controller node.

Subscribes to /task_manifest, parses the intent JSON,
and drives turtlesim to the target pose using P-control.
Supports:
  - Coordinate locations: "x:5,y:3"
  - Named locations (mapped to predefined coords)
"""

import json
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose


# Predefined named locations (turtlesim canvas is ~11x11)
NAMED_LOCATIONS = {
    "center":  (5.5, 5.5),
    "home":    (1.0, 1.0),
    "kitchen": (9.0, 9.0),
    "table":   (7.0, 3.0),
    "shelf":   (3.0, 8.0),
    "door":    (1.0, 5.5),
}

# Control parameters
LINEAR_GAIN = 1.5
ANGULAR_GAIN = 6.0
GOAL_TOLERANCE = 0.15
CONTROL_RATE = 20  # Hz


class TurtleController(Node):
    """Drives turtlesim based on NLP intent from /task_manifest."""

    def __init__(self):
        super().__init__('turtle_controller')

        # State
        self._pose = None
        self._goal = None
        self._moving = False

        # Subscribers
        self.create_subscription(
            String, '/task_manifest', self._on_manifest, 10)
        self.create_subscription(
            Pose, '/turtle1/pose', self._on_pose, 10)

        # Publisher
        self._cmd_pub = self.create_publisher(
            Twist, '/turtle1/cmd_vel', 10)

        # Control loop timer
        self._timer = self.create_timer(
            1.0 / CONTROL_RATE, self._control_loop)

        self.get_logger().info('Turtle controller ready')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_pose(self, msg: Pose):
        self._pose = msg

    def _on_manifest(self, msg: String):
        try:
            intent = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f'Invalid JSON: {msg.data}')
            return

        action = intent.get('action', 'unknown')
        location = intent.get('location')

        if action == 'unknown':
            self.get_logger().warn('Unknown action — ignoring')
            return

        if action == 'move' and location:
            goal = self._parse_location(location)
            if goal:
                self._goal = goal
                self._moving = True
                self.get_logger().info(
                    f'Moving to ({goal[0]:.1f}, {goal[1]:.1f})')
            else:
                self.get_logger().warn(
                    f'Cannot parse location: {location!r}')
        else:
            self.get_logger().info(
                f'Action "{action}" received — '
                f'target={intent.get("target")}, '
                f'location={location}  (no movement)')

    # ------------------------------------------------------------------
    # Location parsing
    # ------------------------------------------------------------------

    def _parse_location(self, loc: str) -> tuple | None:
        """Parse 'x:5,y:3' or named location to (x, y) tuple."""
        loc = loc.lower().strip()

        # Check named locations first
        if loc in NAMED_LOCATIONS:
            return NAMED_LOCATIONS[loc]

        # Parse coordinate format: "x:5,y:3"
        if 'x:' in loc and 'y:' in loc:
            try:
                parts = loc.split(',')
                x = float(parts[0].split(':')[1])
                y = float(parts[1].split(':')[1])
                # Clamp to turtlesim canvas (0.5–10.5)
                x = max(0.5, min(10.5, x))
                y = max(0.5, min(10.5, y))
                return (x, y)
            except (ValueError, IndexError):
                return None

        return None

    # ------------------------------------------------------------------
    # P-Controller
    # ------------------------------------------------------------------

    def _control_loop(self):
        if not self._moving or self._pose is None or self._goal is None:
            return

        dx = self._goal[0] - self._pose.x
        dy = self._goal[1] - self._pose.y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < GOAL_TOLERANCE:
            # Stop
            self._cmd_pub.publish(Twist())
            self._moving = False
            self.get_logger().info(
                f'Reached goal ({self._goal[0]:.1f}, {self._goal[1]:.1f})')
            self._goal = None
            return

        # Desired heading
        desired_theta = math.atan2(dy, dx)
        angle_error = desired_theta - self._pose.theta

        # Normalize angle to [-pi, pi]
        angle_error = math.atan2(math.sin(angle_error),
                                 math.cos(angle_error))

        cmd = Twist()
        cmd.linear.x = LINEAR_GAIN * distance
        cmd.angular.z = ANGULAR_GAIN * angle_error

        # Clamp linear speed
        cmd.linear.x = min(cmd.linear.x, 3.0)

        self._cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = TurtleController()
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
