#!/usr/bin/env python3
"""Watch a GUIDE camera topic in a window.

Built on ``ROS2Camera``, so what you see is what a policy sees: the same discovery of
raw ``sensor_msgs/Image`` versus H.264 ``CompressedImage``, the same decode, the same
resize and centre-crop to the dataset resolution. If this window is black or stuttering,
so is the policy's input -- which is the point of watching it here rather than in rqt,
whose compressed transport only knows jpeg/png and cannot decode the h264 stream.

Tk and PIL rather than ``cv2.imshow``: the venv has ``opencv-python-headless``
(``GUI: NONE`` in its build information), so imshow raises "The function is not
implemented" no matter what display is available. Tk and PIL are both already there.

Needs the ROS environment and the venv that has PyAV:

    source /opt/ros/jazzy/setup.zsh
    source ~/ros2_ws/install/setup.zsh
    ~/ros2_ws/.venv/bin/python guide_core/scripts/view_camera.py cam_top

q or Esc closes the window.
"""

import argparse
import tkinter as tk

import rclpy
from irob_lerobot_ros.config import ROS2CameraConfig
from irob_lerobot_ros.ros2camera import ROS2Camera
from PIL import Image, ImageTk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("topic", help="Camera topic, relative to --namespace, e.g. cam_top")
    parser.add_argument("--namespace", default="/Sim_0/Scene_0")
    # The dataset resolution: ROS2Camera crops to it, so a mismatch here shows a
    # different framing than the one the policy is fed.
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0, help="Window refresh rate.")
    args = parser.parse_args()

    camera = ROS2Camera(
        ROS2CameraConfig(
            namespace=args.namespace,
            topic=args.topic,
            frame_id=f"view_{args.topic}",
            width=args.width,
            height=args.height,
        )
    )
    camera.connect()

    root = tk.Tk()
    root.title(f"{args.namespace}/{args.topic}")
    label = tk.Label(root)
    label.pack()
    for sequence in ("<Escape>", "q"):
        root.bind(sequence, lambda _event: root.destroy())

    period_ms = max(1, int(1000 / args.fps))

    def refresh() -> None:
        # async_read, not read: read() blocks for five seconds when the stream is quiet,
        # which would freeze the window instead of showing that it is quiet.
        image = camera.async_read(timeout_ms=0)
        if image is not None:
            # ROS2Camera hands back RGB, which is what the policy consumes and what PIL
            # expects -- no channel swap, unlike cv2.
            photo = ImageTk.PhotoImage(Image.fromarray(image))
            label.configure(image=photo)
            label.image = photo  # Tk keeps no reference of its own; without this it blanks
        root.after(period_ms, refresh)

    root.after(0, refresh)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            camera.disconnect()


if __name__ == "__main__":
    main()
