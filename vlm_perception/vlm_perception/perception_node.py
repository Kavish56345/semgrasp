"""
VLM Perception Node — GroundingDINO + MobileSAM + Depth Fusion.

Subscribes:
    /target_object                              (std_msgs/String)
    /camera/color/image_raw                     (sensor_msgs/Image)
    /camera/aligned_depth_to_color/image_raw    (sensor_msgs/Image)
    /camera/color/camera_info                   (sensor_msgs/CameraInfo)

Publishes:
    /target_mask        (sensor_msgs/Image)
    /target_pointcloud  (sensor_msgs/PointCloud2)
    /target_centroid    (geometry_msgs/PointStamped)
    /annotated_image    (sensor_msgs/Image)       # debug visualization

All outputs in frame: camera_color_optical_frame
"""

import os
import sys
import time
import struct

import numpy as np
import torch
import cv2

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

# ── Resolve model paths relative to workspace root ────────────────────
# Assumes workspace is ~/semgrasp (adjust if needed)
_WS = os.path.expanduser("~/semgrasp")

from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor


class PerceptionNode(Node):
    """Perception node: text-prompted detection → segmentation → 3D point cloud."""

    def __init__(self):
        super().__init__("vlm_perception")

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter("box_threshold", 0.4)
        self.declare_parameter("text_threshold", 0.25)
        self.declare_parameter("process_rate_hz", 5.0)  # increased due to skipping
        self.declare_parameter("depth_scale", 0.001)  # D415 default: mm → m
        self.declare_parameter("frame_id", "camera_color_optical_frame")

        self._box_thresh = self.get_parameter("box_threshold").value
        self._text_thresh = self.get_parameter("text_threshold").value
        self._rate = self.get_parameter("process_rate_hz").value
        self._depth_scale = self.get_parameter("depth_scale").value
        self._frame_id = self.get_parameter("frame_id").value

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Device: {self._device}")

        # ── Load models ───────────────────────────────────────────────
        self.get_logger().info("Loading YOLO-World-S...")
        t0 = time.time()
        self._yolo = YOLO("yolov8s-worldv2.pt")
        self._yolo.to(self._device)
        self.get_logger().info(f"YOLO-World-S loaded in {time.time()-t0:.2f}s")

        self.get_logger().info("Loading MobileSAM...")
        t0 = time.time()
        _WS = os.path.expanduser("~/semgrasp")
        sam_checkpoint = os.path.join(_WS, "MobileSAM/weights/mobile_sam.pt")
        sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint)
        sam.to(self._device)
        sam.eval()
        self._predictor = SamPredictor(sam)
        self.get_logger().info(f"MobileSAM loaded in {time.time()-t0:.2f}s")

        # ── Warmup (first inference is always slow) ───────────────────
        self.get_logger().info("Warmup...")
        with torch.no_grad():
            _ = torch.zeros((1, 3, 480, 640), device=self._device)
        self.get_logger().info("Warmup done")

        # ── State ─────────────────────────────────────────────────────
        self._bridge = CvBridge()
        self._target_text = ""           # current text prompt
        self._color_image = None         # latest RGB (numpy BGR)
        self._depth_image = None         # latest aligned depth (numpy uint16)
        self._camera_info = None         # latest CameraInfo
        self._color_header = None        # header from latest color msg
        
        # Counters and Stability state
        self._color_count = 0
        self._depth_count = 0
        self._info_count = 0
        
        self._frame_count = 0
        self._last_box_xyxy_resized = None # stored as [x1, y1, x2, y2] in 640x480
        self._last_centroid = None         # stored as [x, y, z] np.array
        self._last_conf = 0.0
        self._fallback_count = 0           # frames since last valid detection

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(
            String, "/target_object", self._on_target, 10
        )
        self.create_subscription(
            Image, "/camera/camera/color/image_raw", self._on_color, 10
        )
        self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw", self._on_depth, 10
        )
        self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info", self._on_cam_info, 10
        )

        # ── Publishers ────────────────────────────────────────────────
        self._pub_mask = self.create_publisher(Image, "/target_mask", 10)
        self._pub_cloud = self.create_publisher(PointCloud2, "/target_pointcloud", 10)
        self._pub_centroid = self.create_publisher(PointStamped, "/target_centroid", 10)
        self._pub_annotated = self.create_publisher(Image, "/annotated_image", 10)

        # ── Rate-controlled processing timer ──────────────────────────
        period = 1.0 / self._rate
        self.create_timer(period, self._process)

        self.get_logger().info(
            f"Perception node ready  [rate={self._rate}Hz, "
            f"box_thresh={self._box_thresh}, frame={self._frame_id}]"
        )

    # ── Callbacks: just store latest data ─────────────────────────────

    def _on_target(self, msg: String):
        text = msg.data.strip().lower()
        if text != self._target_text:
            self.get_logger().info(f'Target object set: "{text}"')
            self._target_text = text
            # Update YOLO-World classes for open-vocabulary detection
            self._yolo.set_classes([text])

    def _on_color(self, msg: Image):
        self._color_count += 1
        # if self._color_count % 10 == 0:
        #     self.get_logger().info(f"Received {self._color_count} color frames", throttle_duration_sec=5.0)
        self._color_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._color_header = msg.header

    def _on_depth(self, msg: Image):
        self._depth_count += 1
        # if self._depth_count % 10 == 0:
        #     self.get_logger().info(f"Received {self._depth_count} depth frames", throttle_duration_sec=5.0)
        self._depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")

    def _on_cam_info(self, msg: CameraInfo):
        self._info_count += 1
        # if self._info_count % 10 == 0:
        #     self.get_logger().info(f"Received {self._info_count} cam_info msgs", throttle_duration_sec=5.0)
        self._camera_info = msg

    # ── Main processing loop (rate-controlled) ────────────────────────

    def _process(self):
        # Gate: need target + color + depth + intrinsics
        if not self._target_text:
            return
        
        missing = []
        if self._color_image is None: missing.append("color")
        if self._depth_image is None: missing.append("depth")
        if self._camera_info is None: missing.append("cam_info")
        
        if missing:
            self.get_logger().info(f"Waiting for topics: {', '.join(missing)}", throttle_duration_sec=5.0)
            return

        t_start = time.time()

        color = self._color_image.copy()
        depth = self._depth_image.copy()
        text = self._target_text

        # ── Resize to 640×480 for speed ───────────────────────────────
        h_orig, w_orig = color.shape[:2]
        color_resized = cv2.resize(color, (640, 480))

        # ── YOLO-World-S detection ────────────────────────────────────
        image_rgb = cv2.cvtColor(color_resized, cv2.COLOR_BGR2RGB)
 
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(self._device == "cuda")):
                # Run YOLO-World-S inference
                results = self._yolo.predict(
                    image_rgb,
                    conf=self._box_thresh,
                    imgsz=640,
                    verbose=False
                )
 
                if len(results[0].boxes) > 0:
                    # Success: Use top-1 box
                    self._fallback_count = 0
                    box_xyxy = results[0].boxes.xyxy[0].cpu().numpy()
                    self._last_box_xyxy_resized = box_xyxy
                    self._last_conf = float(results[0].boxes.conf[0])
                else:
                    # Detection failed: Try fallback
                    if self._last_box_xyxy_resized is not None and self._fallback_count < 3:
                        self._fallback_count += 1
                        self.get_logger().warn(f'Detection lost. Fallback ({self._fallback_count}/3)', throttle_duration_sec=1.0)
                    else:
                        self.get_logger().warn(f'No detection for "{text}"', throttle_duration_sec=5.0)
                        self._last_box_xyxy_resized = None
                        return
 
        det_time = time.time() - t_start

        # ── MobileSAM segmentation (on resized image) ─────────────────
        t_seg = time.time()
        box_for_sam = self._last_box_xyxy_resized
        self._predictor.set_image(image_rgb)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(self._device == "cuda")):
                masks, scores, _ = self._predictor.predict(
                    box=box_for_sam,
                    multimask_output=False,
                )
        seg_time = time.time() - t_seg

        mask_resized = masks[0].astype(np.uint8)  # (480, 640), values 0 or 1

        # ── Scale mask back to original resolution ────────────────────
        mask_full = cv2.resize(
            mask_resized, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST
        )

        # ── Publish mask ──────────────────────────────────────────────
        mask_msg = self._bridge.cv2_to_imgmsg(mask_full * 255, encoding="mono8")
        mask_msg.header = self._make_header()
        self._pub_mask.publish(mask_msg)

        # ── Depth fusion: mask × aligned depth ────────────────────────
        # Resize depth to match original color if needed
        if depth.shape[:2] != (h_orig, w_orig):
            depth = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        masked_depth = depth.copy()
        masked_depth[mask_full == 0] = 0

        # ── Point cloud from masked depth using intrinsics ────────────
        fx = self._camera_info.k[0]
        fy = self._camera_info.k[4]
        cx_cam = self._camera_info.k[2]
        cy_cam = self._camera_info.k[5]

        # Get masked pixel coordinates
        vs, us = np.where(mask_full > 0)
        zs = masked_depth[vs, us].astype(np.float32) * self._depth_scale

        # Filter out zero/invalid depth
        valid = zs > 0.01  # at least 1cm
        vs, us, zs = vs[valid], us[valid], zs[valid]

        if len(zs) == 0:
            self.get_logger().warn("No valid depth in masked region", throttle_duration_sec=5.0)
            return

        # Compute 3D points in camera_color_optical_frame
        xs = (us.astype(np.float32) - cx_cam) * zs / fx
        ys = (vs.astype(np.float32) - cy_cam) * zs / fy
        points_3d = np.stack([xs, ys, zs], axis=-1)  # (N, 3)

        # ── Publish PointCloud2 ───────────────────────────────────────
        cloud_msg = self._make_pointcloud2(points_3d)
        self._pub_cloud.publish(cloud_msg)

        # ── Compute and publish centroid ──────────────────────────────
        centroid = points_3d.mean(axis=0)

        self._last_centroid = centroid
        current_z = float(centroid[2])

        centroid_msg = PointStamped()
        centroid_msg.header = self._make_header()
        centroid_msg.point.x = float(centroid[0])
        centroid_msg.point.y = float(centroid[1])
        centroid_msg.point.z = current_z
        self._pub_centroid.publish(centroid_msg)

        # ── Publish annotated debug image ─────────────────────────────
        annotated = color.copy()
        # Scale box to original resolution
        sx, sy = w_orig / 640.0, h_orig / 480.0
        x1_o = int(self._last_box_xyxy_resized[0] * sx)
        y1_o = int(self._last_box_xyxy_resized[1] * sy)
        x2_o = int(self._last_box_xyxy_resized[2] * sx)
        y2_o = int(self._last_box_xyxy_resized[3] * sy)
        cv2.rectangle(annotated, (x1_o, y1_o), (x2_o, y2_o), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{text} {self._last_conf:.2f}",
            (x1_o, y1_o - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        mask_overlay = np.zeros_like(annotated)
        mask_overlay[:, :, 1] = mask_full * 255
        annotated = cv2.addWeighted(annotated, 0.7, mask_overlay, 0.3, 0)

        ann_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        ann_msg.header = self._make_header()
        self._pub_annotated.publish(ann_msg)

        total_time = time.time() - t_start
        self.get_logger().info(
            f'[{text}] YOLO={det_time:.3f}s SAM={seg_time:.3f}s Total={total_time:.3f}s '
            f'conf={self._last_conf:.2f} centroid=({centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f})'
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _make_header(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self._frame_id
        return header

    def _make_pointcloud2(self, points: np.ndarray) -> PointCloud2:
        """Create PointCloud2 from Nx3 float32 array."""
        msg = PointCloud2()
        msg.header = self._make_header()
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = True
        msg.is_bigendian = False
        msg.point_step = 12  # 3 × float32
        msg.row_step = msg.point_step * msg.width

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg.data = points.astype(np.float32).tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
