import os
import csv
import cv2
import numpy as np
import pyrealsense2 as rs

# ========= CONFIG =========
BAG_FILE = "bag_examples/toma2_d435i.bag"
OUT_DIR = "salida_bag"
EXPORT_PNG_FRAMES = False
COLOR_FPS = 30
DEPTH_FPS = 30
IR_FPS = 30
# ==========================

os.makedirs(OUT_DIR, exist_ok=True)

color_frames_dir = os.path.join(OUT_DIR, "color_frames")
depth_frames_dir = os.path.join(OUT_DIR, "depth_frames")
ir_frames_dir = os.path.join(OUT_DIR, "ir_frames")

if EXPORT_PNG_FRAMES:
    os.makedirs(color_frames_dir, exist_ok=True)
    os.makedirs(depth_frames_dir, exist_ok=True)
    os.makedirs(ir_frames_dir, exist_ok=True)

imu_csv = os.path.join(OUT_DIR, "imu.csv")
color_video_path = os.path.join(OUT_DIR, "color.mp4")
depth_video_path = os.path.join(OUT_DIR, "depth_vis.mp4")
ir_video_path = os.path.join(OUT_DIR, "infrared.mp4")

pipeline = rs.pipeline()
config = rs.config()

rs.config.enable_device_from_file(config, BAG_FILE, repeat_playback=False)

profile = pipeline.start(config)

playback = profile.get_device().as_playback()
playback.set_real_time(False)

colorizer = rs.colorizer()

color_writer = None
depth_writer = None
ir_writer = None

frame_idx = 0
imu_count = 0

with open(imu_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["type", "frame_number", "timestamp_ms", "x", "y", "z"])

    try:
        while True:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            infra_frame = frames.get_infrared_frame()

            # -------- COLOR --------
            if color_frame:
                color = np.asanyarray(color_frame.get_data())
                color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

                if color_writer is None:
                    h, w = color_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    color_writer = cv2.VideoWriter(color_video_path, fourcc, COLOR_FPS, (w, h))

                color_writer.write(color_bgr)

                if EXPORT_PNG_FRAMES:
                    cv2.imwrite(
                        os.path.join(color_frames_dir, f"color_{frame_idx:06d}.png"),
                        color_bgr
                    )

            # -------- DEPTH --------
            if depth_frame:
                depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                depth_vis_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)

                if depth_writer is None:
                    h, w = depth_vis_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    depth_writer = cv2.VideoWriter(depth_video_path, fourcc, DEPTH_FPS, (w, h))

                depth_writer.write(depth_vis_bgr)

                if EXPORT_PNG_FRAMES:
                    depth_raw = np.asanyarray(depth_frame.get_data())
                    cv2.imwrite(
                        os.path.join(depth_frames_dir, f"depth_{frame_idx:06d}.png"),
                        depth_raw
                    )

            # -------- INFRARED --------
            if infra_frame:
                ir = np.asanyarray(infra_frame.get_data())  # normalmente uint8, 1 canal

                if ir_writer is None:
                    h, w = ir.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    ir_writer = cv2.VideoWriter(ir_video_path, fourcc, IR_FPS, (w, h), False)

                ir_writer.write(ir)

                if EXPORT_PNG_FRAMES:
                    cv2.imwrite(
                        os.path.join(ir_frames_dir, f"ir_{frame_idx:06d}.png"),
                        ir
                    )

            # -------- IMU --------
            for i in range(frames.size()):
                fr = frames[i]

                if fr.is_motion_frame():
                    motion = fr.as_motion_frame().get_motion_data()
                    st = fr.get_profile().stream_type()

                    if st == rs.stream.accel:
                        motion_type = "accel"
                    elif st == rs.stream.gyro:
                        motion_type = "gyro"
                    else:
                        motion_type = "motion"

                    writer.writerow([
                        motion_type,
                        fr.get_frame_number(),
                        fr.get_timestamp(),
                        motion.x,
                        motion.y,
                        motion.z
                    ])
                    imu_count += 1

            frame_idx += 1

    except RuntimeError:
        pass

    finally:
        pipeline.stop()
        if color_writer is not None:
            color_writer.release()
        if depth_writer is not None:
            depth_writer.release()
        if ir_writer is not None:
            ir_writer.release()

print("Listo.")
print(f"Frames de video procesados: {frame_idx}")
print(f"Muestras IMU guardadas: {imu_count}")
print(f"Video color: {color_video_path}")
print(f"Video depth: {depth_video_path}")
print(f"Video infrared: {ir_video_path}")
print(f"IMU CSV: {imu_csv}")