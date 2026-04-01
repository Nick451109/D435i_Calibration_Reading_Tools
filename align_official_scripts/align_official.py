#pip install pyrealsense2 numpy opencv-python

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

def clamp(value, min_v, max_v):
    return max(min_v, min(value, max_v))

def get_option_info(sensor, option):
    if sensor.supports(option):
        r = sensor.get_option_range(option)
        return r.min, r.max, r.step, r.default
    return None

def safe_get(sensor, option, fallback=None):
    try:
        if sensor.supports(option):
            return sensor.get_option(option)
    except:
        pass
    return fallback

def safe_set(sensor, option, value):
    try:
        if sensor.supports(option):
            sensor.set_option(option, value)
            return True
    except Exception as e:
        print(f"No se pudo ajustar {option}: {e}")
    return False

def apply_preset(depth_sensor, preset_value):
    ok = safe_set(depth_sensor, rs.option.visual_preset, float(preset_value))
    if ok:
        preset_names = {
            1: "Default",
            3: "High Accuracy",
            4: "High Density"
        }
        print(f"Preset aplicado: {preset_names.get(preset_value, preset_value)}")
    return ok

def main():
    save_dir = "captures_realsense"
    os.makedirs(save_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    align = rs.align(rs.stream.color)

    depth_scale = depth_sensor.get_depth_scale()
    print("Depth scale:", depth_scale)

    # Rangos disponibles
    laser_info = get_option_info(depth_sensor, rs.option.laser_power)
    gain_info = get_option_info(depth_sensor, rs.option.gain)
    exposure_info = get_option_info(depth_sensor, rs.option.exposure)
    autoexp_info = get_option_info(depth_sensor, rs.option.enable_auto_exposure)

    if laser_info:
        print("Laser power range:", laser_info)
    if gain_info:
        print("Gain range:", gain_info)
    if exposure_info:
        print("Exposure range:", exposure_info)

    # Configuración inicial
    apply_preset(depth_sensor, 3)  # High Accuracy
    safe_set(depth_sensor, rs.option.emitter_enabled, 1)

    if laser_info:
        laser_min, laser_max, laser_step, _ = laser_info
        safe_set(depth_sensor, rs.option.laser_power, laser_max)

    if autoexp_info:
        safe_set(depth_sensor, rs.option.enable_auto_exposure, 1)

    # Filtros
    decimation = rs.decimation_filter()
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    decimation.set_option(rs.option.filter_magnitude, 2)

    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    filters_enabled = True

    print("\nControles:")
    print("  1 = Default preset")
    print("  2 = High Accuracy preset")
    print("  3 = High Density preset")
    print("  +/= = subir laser power")
    print("  - = bajar laser power")
    print("  g = subir gain")
    print("  h = bajar gain")
    print("  e = subir exposure")
    print("  r = bajar exposure")
    print("  a = alternar auto exposure")
    print("  f = alternar filtros")
    print("  c = capturar")
    print("  ESC = salir\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            processed_depth = depth_frame
            if filters_enabled:
                processed_depth = spatial.process(processed_depth)
                processed_depth = temporal.process(processed_depth)
                processed_depth = hole_filling.process(processed_depth)

            aligned_depth = np.asanyarray(processed_depth.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_m = aligned_depth.astype(np.float32) * depth_scale
            min_m = 0.4
            max_m = 2.5
            depth_clipped = np.clip(depth_m, min_m, max_m)
            depth_gray = ((depth_clipped - min_m) / (max_m - min_m) * 255).astype(np.uint8)

            depth_gray_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(color_image, 0.7, depth_gray_bgr, 0.3, 0)

            # Texto de estado
            laser_val = safe_get(depth_sensor, rs.option.laser_power, -1)
            gain_val = safe_get(depth_sensor, rs.option.gain, -1)
            exposure_val = safe_get(depth_sensor, rs.option.exposure, -1)
            autoexp_val = safe_get(depth_sensor, rs.option.enable_auto_exposure, -1)

            status = (
                f"Laser:{laser_val:.1f}  "
                f"Gain:{gain_val:.1f}  "
                f"Exposure:{exposure_val:.1f}  "
                f"AutoExp:{int(autoexp_val) if autoexp_val is not None else -1}  "
                f"Filtros:{'ON' if filters_enabled else 'OFF'}"
            )

            overlay_text = overlay.copy()
            cv2.putText(
                overlay_text,
                status,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Color", color_image)
            cv2.imshow("Depth aligned to Color (Grayscale)", depth_gray)
            cv2.imshow("Overlay", overlay_text)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            elif key == ord('1'):
                apply_preset(depth_sensor, 1)

            elif key == ord('2'):
                apply_preset(depth_sensor, 3)

            elif key == ord('3'):
                apply_preset(depth_sensor, 4)

            elif key in (ord('+'), ord('=')):
                if laser_info:
                    laser_min, laser_max, laser_step, _ = laser_info
                    current = safe_get(depth_sensor, rs.option.laser_power, laser_min)
                    new_value = clamp(current + max(laser_step, 1), laser_min, laser_max)
                    safe_set(depth_sensor, rs.option.laser_power, new_value)
                    print(f"Laser power -> {new_value}")

            elif key == ord('-'):
                if laser_info:
                    laser_min, laser_max, laser_step, _ = laser_info
                    current = safe_get(depth_sensor, rs.option.laser_power, laser_min)
                    new_value = clamp(current - max(laser_step, 1), laser_min, laser_max)
                    safe_set(depth_sensor, rs.option.laser_power, new_value)
                    print(f"Laser power -> {new_value}")

            elif key == ord('g'):
                if gain_info:
                    gain_min, gain_max, gain_step, _ = gain_info
                    current = safe_get(depth_sensor, rs.option.gain, gain_min)
                    new_value = clamp(current + max(gain_step, 1), gain_min, gain_max)
                    safe_set(depth_sensor, rs.option.gain, new_value)
                    print(f"Gain -> {new_value}")

            elif key == ord('h'):
                if gain_info:
                    gain_min, gain_max, gain_step, _ = gain_info
                    current = safe_get(depth_sensor, rs.option.gain, gain_min)
                    new_value = clamp(current - max(gain_step, 1), gain_min, gain_max)
                    safe_set(depth_sensor, rs.option.gain, new_value)
                    print(f"Gain -> {new_value}")

            elif key == ord('e'):
                if exposure_info:
                    exp_min, exp_max, exp_step, _ = exposure_info
                    current = safe_get(depth_sensor, rs.option.exposure, exp_min)
                    new_value = clamp(current + max(exp_step, 1), exp_min, exp_max)
                    safe_set(depth_sensor, rs.option.exposure, new_value)
                    print(f"Exposure -> {new_value}")

            elif key == ord('r'):
                if exposure_info:
                    exp_min, exp_max, exp_step, _ = exposure_info
                    current = safe_get(depth_sensor, rs.option.exposure, exp_min)
                    new_value = clamp(current - max(exp_step, 1), exp_min, exp_max)
                    safe_set(depth_sensor, rs.option.exposure, new_value)
                    print(f"Exposure -> {new_value}")

            elif key == ord('a'):
                if autoexp_info:
                    current = safe_get(depth_sensor, rs.option.enable_auto_exposure, 1)
                    new_value = 0 if int(current) == 1 else 1
                    safe_set(depth_sensor, rs.option.enable_auto_exposure, new_value)
                    print(f"Auto exposure -> {'ON' if new_value == 1 else 'OFF'}")

            elif key == ord('f'):
                filters_enabled = not filters_enabled
                print(f"Filtros -> {'ON' if filters_enabled else 'OFF'}")

            elif key == ord('c'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                color_path = os.path.join(save_dir, f"color_{timestamp}.png")
                depth_gray_path = os.path.join(save_dir, f"depth_gray_{timestamp}.png")
                overlay_path = os.path.join(save_dir, f"overlay_{timestamp}.png")
                depth_raw_path = os.path.join(save_dir, f"depth_raw_{timestamp}.npy")

                cv2.imwrite(color_path, color_image)
                cv2.imwrite(depth_gray_path, depth_gray)
                cv2.imwrite(overlay_path, overlay_text)
                np.save(depth_raw_path, aligned_depth)

                print(f"Captura guardada:")
                print(f"  {color_path}")
                print(f"  {depth_gray_path}")
                print(f"  {overlay_path}")
                print(f"  {depth_raw_path}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()