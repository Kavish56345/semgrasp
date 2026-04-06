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

GROUNDING_CONFIG = os.path.join(
    _WS, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_WEIGHTS = os.path.join(
    _WS, "GroundingDINO/weights/groundingdino_swint_ogc.pth"
)
SAM_CHECKPOINT = os.path.join(
    _WS, "MobileSAM/weights/mobile_sam.pt"
)

# Add to path so imports work inside venv
sys.path.insert(0, os.path.join(_WS, "GroundingDINO"))
sys.path.insert(0, os.path.join(_WS, "MobileSAM"))

from groundingdino.util.inference import load_model, load_image, predict
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
        self.get_logger().info("Loading GroundingDINO...")
        t0 = time.time()
        self._grounding = load_model(GROUNDING_CONFIG, GROUNDING_WEIGHTS)
        self.get_logger().info(f"GroundingDINO loaded in {time.time()-t0:.2f}s")

        self.get_logger().info("Loading MobileSAM...")
        t0 = time.time()
        sam = sam_model_registry["vit_t"](checkpoint=SAM_CHECKPOINT)
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
        self._last_phrases = None
        self._last_logits = None
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

        # ── GroundingDINO detection ───────────────────────────────────
        # load_image expects a file path; we use raw numpy instead
        image_rgb = cv2.cvtColor(color_resized, cv2.COLOR_BGR2RGB)

        # Prepare image tensor the way GroundingDINO expects
        import groundingdino.datasets.transforms as T
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image_rgb)
        image_tensor, _ = transform(pil_image, None)

        self._frame_count += 1
        run_detection = (self._frame_count % 3 == 0) or (self._last_box_xyxy_resized is None)

        # ── Detection / Tracking Block ────────────────────────────────
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(self._device == "cuda")):
                if run_detection:
                    # Execute heavy GroundingDINO detection
                    boxes, logits, phrases = predict(
                        model=self._grounding,
                        image=image_tensor,
                        caption=text,
                        box_threshold=self._box_thresh,
                        text_threshold=self._text_thresh,
                    )

                    if len(boxes) > 0 and logits[0] >= self._box_thresh:
                        # Success: Use new box
                        self._fallback_count = 0
                        box = boxes[0]
                        cx, cy, bw, bh = box
                        x1_r = (cx - bw / 2) * 640
                        y1_r = (cy - bh / 2) * 480
                        x2_r = (cx + bw / 2) * 640
                        y2_r = (cy + bh / 2) * 480
                        self._last_box_xyxy_resized = np.array([x1_r, y1_r, x2_r, y2_r])
                        self._last_logits = logits[:1]
                        self._last_phrases = phrases[:1]
                    else:
                        # Detection failed: Try fallback
                        if self._last_box_xyxy_resized is not None and self._fallback_count < 3:
                            self._fallback_count += 1
                            self.get_logger().warn(f'Detection lost. Fallback ({self._fallback_count}/3)', throttle_duration_sec=1.0)
                        else:
                            self.get_logger().warn(f'No detection for "{text}"', throttle_duration_sec=5.0)
                            self._last_box_xyxy_resized = None
                            return
                else:
                    # Skipping DINO: Use previous box
                    if self._last_box_xyxy_resized is None:
                        return # safety
                    # self.get_logger().info("Skipping detection (reuse last box)", throttle_duration_sec=2.0)

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

        # ── Stability Check: Adaptive Distance Threshold ──────────────
        current_z = float(centroid[2])
        if self._last_centroid is not None:
            dist = np.linalg.norm(centroid - self._last_centroid)
            adaptive_thresh = 0.1 + 0.2 * current_z
            if dist > adaptive_thresh:
                self.get_logger().warn(
                    f"Centroid jump detected: {dist:.3f}m > limit {adaptive_thresh:.3f}m. Rejecting frame.",
                    throttle_duration_sec=1.0
                )
                return
        
        self._last_centroid = centroid

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
            f"{self._last_phrases[0]} {self._last_logits[0]:.2f}",
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
            f'[{text}] conf={self._last_logits[0]:.2f} pts={len(points_3d)} '
            f'centroid=({centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f}) '
            f'det={det_time:.2f}s seg={seg_time:.2f}s total={total_time:.2f}s'
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
