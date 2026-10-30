#!/usr/bin/env python3
"""Standalone browser 3D view of VR retargeted keybody positions."""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


IGRIS_C_EDGES = (
    ("Link_Waist_Yaw", "Link_Waist_Roll"),
    ("Link_Waist_Roll", "Link_Waist_Pitch"),
    ("Link_Waist_Pitch", "Link_Hip_Pitch_Left"),
    ("Link_Hip_Pitch_Left", "Link_Hip_Roll_Left"),
    ("Link_Hip_Roll_Left", "Link_Hip_Yaw_Left"),
    ("Link_Hip_Yaw_Left", "Link_Knee_Pitch_Left"),
    ("Link_Knee_Pitch_Left", "Link_Ankle_Pitch_Left"),
    ("Link_Ankle_Pitch_Left", "Link_Ankle_Roll_Left"),
    ("Link_Waist_Pitch", "Link_Hip_Pitch_Right"),
    ("Link_Hip_Pitch_Right", "Link_Hip_Roll_Right"),
    ("Link_Hip_Roll_Right", "Link_Hip_Yaw_Right"),
    ("Link_Hip_Yaw_Right", "Link_Knee_Pitch_Right"),
    ("Link_Knee_Pitch_Right", "Link_Ankle_Pitch_Right"),
    ("Link_Ankle_Pitch_Right", "Link_Ankle_Roll_Right"),
    ("Link_Waist_Pitch", "Link_Shoulder_Pitch_Left"),
    ("Link_Shoulder_Pitch_Left", "Link_Shoulder_Roll_Left"),
    ("Link_Shoulder_Roll_Left", "Link_Shoulder_Yaw_Left"),
    ("Link_Shoulder_Yaw_Left", "Link_Elbow_Pitch_Left"),
    ("Link_Elbow_Pitch_Left", "Link_Wrist_Yaw_Left"),
    ("Link_Wrist_Yaw_Left", "Link_Wrist_Roll_Left"),
    ("Link_Wrist_Roll_Left", "Link_Wrist_Pitch_Left"),
    ("Link_Wrist_Pitch_Left", "Left_Hand"),
    ("Link_Waist_Pitch", "Link_Shoulder_Pitch_Right"),
    ("Link_Shoulder_Pitch_Right", "Link_Shoulder_Roll_Right"),
    ("Link_Shoulder_Roll_Right", "Link_Shoulder_Yaw_Right"),
    ("Link_Shoulder_Yaw_Right", "Link_Elbow_Pitch_Right"),
    ("Link_Elbow_Pitch_Right", "Link_Wrist_Yaw_Right"),
    ("Link_Wrist_Yaw_Right", "Link_Wrist_Roll_Right"),
    ("Link_Wrist_Roll_Right", "Link_Wrist_Pitch_Right"),
    ("Link_Wrist_Pitch_Right", "Right_Hand"),
    ("Link_Waist_Pitch", "Link_Neck_Yaw"),
    ("Link_Neck_Yaw", "Link_Neck_Pitch"),
)

BODY_CHAINS = (
    ("base_link", "Link_Waist_Yaw", "Link_Waist_Roll", "Link_Waist_Pitch"),
    ("base_link", "Link_Waist_Pitch", "Link_Hip_Pitch_Left", "Link_Hip_Roll_Left",
     "Link_Hip_Yaw_Left", "Link_Knee_Pitch_Left", "Link_Ankle_Pitch_Left", "Link_Ankle_Roll_Left"),
    ("base_link", "Link_Waist_Pitch", "Link_Hip_Pitch_Right", "Link_Hip_Roll_Right",
     "Link_Hip_Yaw_Right", "Link_Knee_Pitch_Right", "Link_Ankle_Pitch_Right", "Link_Ankle_Roll_Right"),
    ("base_link", "Link_Waist_Pitch", "Link_Shoulder_Pitch_Left", "Link_Shoulder_Roll_Left",
     "Link_Shoulder_Yaw_Left", "Link_Elbow_Pitch_Left", "Link_Wrist_Yaw_Left",
     "Link_Wrist_Roll_Left", "Link_Wrist_Pitch_Left", "Left_Hand"),
    ("base_link", "Link_Waist_Pitch", "Link_Shoulder_Pitch_Right", "Link_Shoulder_Roll_Right",
     "Link_Shoulder_Yaw_Right", "Link_Elbow_Pitch_Right", "Link_Wrist_Yaw_Right",
     "Link_Wrist_Roll_Right", "Link_Wrist_Pitch_Right", "Right_Hand"),
    ("base_link", "Link_Waist_Pitch", "Link_Neck_Yaw", "Link_Neck_Pitch"),
)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VR Motion Viewer</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; background: #101316; color: #e6edf3; font-family: system-ui, sans-serif; }
    body { overflow: hidden; }
    #hud { position: fixed; left: 16px; top: 12px; z-index: 2; font-size: 14px; line-height: 1.45; color: #d5dde5; }
    #hud b { color: #ffffff; }
    #canvas { display: block; width: 100vw; height: 100vh; touch-action: none; }
    #empty { position: fixed; inset: 0; display: grid; place-items: center; color: #8b949e; font-size: 18px; pointer-events: none; }
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <div id="hud">
    <div><b>VR Motion Viewer</b></div>
    <div id="status">connecting...</div>
    <div>drag: rotate, wheel/pinch: zoom · floor grid: 0.5 m</div>
  </div>
  <div id="empty">waiting for keybody positions</div>
  <script>
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const emptyEl = document.getElementById("empty");
    let latest = null;
    let yaw = -0.75;
    let pitch = -0.35;
    let zoom = 190;
    const pointers = new Map();
    let lastDrag = null;
    let lastPinchDistance = null;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(window.innerWidth * dpr);
      canvas.height = Math.floor(window.innerHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    window.addEventListener("resize", resize);
    resize();

    function clampZoom() {
      zoom = Math.max(60, Math.min(600, zoom));
    }

    function pointerDistance() {
      const pts = Array.from(pointers.values());
      if (pts.length < 2) return null;
      return Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
    }

    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 1) {
        lastDrag = { x: e.clientX, y: e.clientY };
        lastPinchDistance = null;
      } else if (pointers.size === 2) {
        lastDrag = null;
        lastPinchDistance = pointerDistance();
      }
    });

    canvas.addEventListener("pointermove", (e) => {
      if (!pointers.has(e.pointerId)) return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 1 && lastDrag) {
        yaw += (e.clientX - lastDrag.x) * 0.008;
        pitch -= (e.clientY - lastDrag.y) * 0.008;
        pitch = Math.max(-1.45, Math.min(1.45, pitch));
        lastDrag = { x: e.clientX, y: e.clientY };
        return;
      }
      if (pointers.size === 2) {
        const distance = pointerDistance();
        if (distance !== null && lastPinchDistance !== null && lastPinchDistance > 0) {
          zoom *= distance / lastPinchDistance;
          clampZoom();
        }
        lastPinchDistance = distance;
      }
    });

    function releasePointer(e) {
      pointers.delete(e.pointerId);
      if (pointers.size === 1) {
        const p = Array.from(pointers.values())[0];
        lastDrag = { x: p.x, y: p.y };
        lastPinchDistance = null;
      } else {
        lastDrag = null;
        lastPinchDistance = null;
      }
    }

    canvas.addEventListener("pointerup", releasePointer);
    canvas.addEventListener("pointercancel", releasePointer);
    canvas.addEventListener("lostpointercapture", releasePointer);

    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      zoom *= Math.exp(-e.deltaY * 0.001);
      clampZoom();
    }, { passive: false });

    const source = new EventSource("/events");
    source.onmessage = (event) => {
      latest = JSON.parse(event.data);
      const age = latest.age_ms == null ? "unknown" : `${latest.age_ms} ms`;
      const src = latest.source || "unknown";
      statusEl.textContent = `frames ${latest.frame_idx ?? "-"} | joints ${latest.points.length} | age ${age} | ${src}`;
    };
    source.onerror = () => {
      statusEl.textContent = "disconnected";
    };

    function rotatePoint(p, center) {
      let x = p[0] - center[0];
      let y = p[1] - center[1];
      let z = p[2] - center[2];
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = cy * x + sy * y;
      const y1 = -sy * x + cy * y;
      const z1 = z;
      const y2 = cp * y1 - sp * z1;
      const z2 = sp * y1 + cp * z1;
      return [x1, y2, z2];
    }

    function project(p) {
      return [window.innerWidth * 0.5 + p[0] * zoom, window.innerHeight * 0.57 - p[2] * zoom];
    }

    function drawFloorGrid(center) {
      const step = 0.5;
      const radius = 3;
      const x0 = Math.floor(center[0] / step) * step;
      const y0 = Math.floor(center[1] / step) * step;
      const xMin = x0 - radius, xMax = x0 + radius;
      const yMin = y0 - radius, yMax = y0 + radius;
      ctx.lineWidth = 1;
      for (let i = -6; i <= 6; i++) {
        const x = x0 + i * step;
        const y = y0 + i * step;
        const xLine = [project(rotatePoint([x, yMin, 0], center)),
                       project(rotatePoint([x, yMax, 0], center))];
        const yLine = [project(rotatePoint([xMin, y, 0], center)),
                       project(rotatePoint([xMax, y, 0], center))];
        ctx.strokeStyle = Math.abs(x) < 1e-6 ? "#6f9b91" :
                          i % 2 === 0 ? "#394752" : "#29343d";
        ctx.beginPath();
        ctx.moveTo(xLine[0][0], xLine[0][1]);
        ctx.lineTo(xLine[1][0], xLine[1][1]);
        ctx.stroke();
        ctx.strokeStyle = Math.abs(y) < 1e-6 ? "#9b746f" :
                          i % 2 === 0 ? "#394752" : "#29343d";
        ctx.beginPath();
        ctx.moveTo(yLine[0][0], yLine[0][1]);
        ctx.lineTo(yLine[1][0], yLine[1][1]);
        ctx.stroke();
      }
    }

    function draw() {
      requestAnimationFrame(draw);
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      if (!latest || latest.points.length === 0) {
        emptyEl.style.display = "grid";
        return;
      }
      emptyEl.style.display = "none";

      const pts = latest.points;
      const center = [0, 0, 0];
      for (const p of pts) {
        center[0] += p[0]; center[1] += p[1]; center[2] += p[2];
      }
      center[0] /= pts.length; center[1] /= pts.length; center[2] /= pts.length;

      drawFloorGrid(center);
      const projected = pts.map((p) => project(rotatePoint(p, center)));
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#e6edf3";
      ctx.lineWidth = 2;
      for (const [a, b] of latest.edges) {
        if (!projected[a] || !projected[b]) continue;
        ctx.beginPath();
        ctx.moveTo(projected[a][0], projected[a][1]);
        ctx.lineTo(projected[b][0], projected[b][1]);
        ctx.stroke();
      }

      for (let i = 0; i < projected.length; i++) {
        const p = projected[i];
        ctx.fillStyle = i === 0 ? "#ffcf33" : "#40a6ff";
        ctx.beginPath();
        ctx.arc(p[0], p[1], 4.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    draw();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/gmr/teleop/retarget_frame")
    parser.add_argument("--ros_domain_id", "--ros-domain-id", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--update_hz", "--update-hz", type=float, default=20.0)
    return parser.parse_args()


def parse_payload(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def first_frame(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


def vector(value: Any, size: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{field} must contain {size} numbers")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} contains a non-finite value")
    return result


def rotate_vector(q: list[float], p: list[float]) -> list[float]:
    w, x, y, z = q
    px, py, pz = p
    tx, ty, tz = 2 * (y * pz - z * py), 2 * (z * px - x * pz), 2 * (x * py - y * px)
    return [px + w * tx + y * tz - z * ty,
            py + w * ty + z * tx - x * tz,
            pz + w * tz + x * ty - y * tx]


def payload_points(payload: dict[str, Any]) -> tuple[list[list[float]], list[str]]:
    names = payload.get("link_body_list")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("link_body_list must be a list of names")
    positions = first_frame(payload.get("keybody_pos_local"))
    if not isinstance(positions, list) or len(positions) != len(names):
        raise ValueError("keybody_pos_local does not match link_body_list")
    local = [vector(point, 3, "keybody_pos_local") for point in positions]
    if "base_link" not in names:
        names = ["base_link", *names]
        local = [[0.0, 0.0, 0.0], *local]
    root_pos = vector(first_frame(payload.get("root_pos")), 3, "root_pos")
    q = vector(first_frame(payload.get("root_rot")), 4, "root_rot")
    norm = math.sqrt(sum(item * item for item in q))
    if norm < 1e-8:
        raise ValueError("root_rot has zero norm")
    q = [item / norm for item in q]
    points = [[a + b for a, b in zip(root_pos, rotate_vector(q, point))] for point in local]
    return points, names


def named_edges(names: list[str]) -> list[tuple[int, int]]:
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    edges = {(name_to_idx[a], name_to_idx[b]) for a, b in IGRIS_C_EDGES
             if a in name_to_idx and b in name_to_idx}
    for chain in BODY_CHAINS:
        if len(chain) > 1 and chain[1] == "Link_Waist_Pitch" and "Link_Waist_Pitch" in name_to_idx:
            chain = chain[1:]
        present = [name_to_idx[name] for name in chain if name in name_to_idx]
        edges.update(zip(present, present[1:]))
    return sorted(edges)


class LatestFrame:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame: dict[str, Any] | None = None
        self.received_at: float | None = None
        self.frames = 0
        self.invalid_frames = 0
        self.last_error = ""

    def set(self, payload: dict[str, Any]) -> None:
        try:
            points, names = payload_points(payload)
        except (ValueError, TypeError, OverflowError) as exc:
            with self.lock:
                self.invalid_frames += 1
                self.last_error = str(exc)
            return
        meta = payload.get("record_meta") if isinstance(payload.get("record_meta"), dict) else {}
        frame = {
            "points": points,
            "names": names,
            "edges": named_edges(names),
            "source": meta.get("source") if isinstance(meta, dict) else None,
            "frame_idx": meta.get("frame_idx") if isinstance(meta, dict) else None,
        }
        with self.lock:
            self.frame = frame
            self.received_at = time.monotonic()
            self.frames += 1
            self.last_error = ""

    def get(self) -> dict[str, Any]:
        with self.lock:
            if self.frame is None:
                result = {"points": [], "names": [], "edges": [], "source": None, "frame_idx": None}
            else:
                result = dict(self.frame)
            result.update(age_ms=None if self.received_at is None else
                          int((time.monotonic() - self.received_at) * 1000),
                          frames=self.frames, invalid_frames=self.invalid_frames,
                          last_error=self.last_error)
            return result


def make_handler(latest: LatestFrame, update_hz: float):
    period = 1.0 / max(1.0, update_hz)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_html()
            elif path == "/events":
                self._send_events()
            elif path == "/api/state":
                body = json.dumps(latest.get(), separators=(",", ":")).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def _send_html(self) -> None:
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            while True:
                data = json.dumps(latest.get(), separators=(",", ":")).encode("utf-8")
                try:
                    self.wfile.write(b"data: " + data + b"\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                time.sleep(period)

    return Handler


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.update_hz) or args.update_hz <= 0:
        raise SystemExit("--update-hz must be positive")
    host = args.host or "127.0.0.1"
    if args.ros_domain_id is not None:
        os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    latest = LatestFrame()

    class RetargetFrameSubscriber(Node):
        def __init__(self):
            super().__init__("igris_c_vr_motion_viewer")
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self.create_subscription(String, args.topic, self._callback, qos)
            self.get_logger().info(
                f"Subscribing to {args.topic} (ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', 'default')})"
            )

        def _callback(self, msg: String) -> None:
            payload = parse_payload(msg.data)
            if payload is None:
                self.get_logger().warning("Failed to parse retarget_frame payload")
                return
            latest.set(payload)

    rclpy.init()
    node = RetargetFrameSubscriber()
    server = ThreadingHTTPServer((host, args.port), make_handler(latest, args.update_hz))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"VR Motion Viewer: http://{host}:{args.port}", flush=True)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
