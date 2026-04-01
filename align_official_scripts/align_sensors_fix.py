# pip install pyrealsense2 numpy opencv-python

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime

SETTINGS_FILE = "../calibration_settings/d435i_outdoor_device_settings.json"


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
    except Exception:
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


def find_d400_device():
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        raise RuntimeError("No se encontró ninguna cámara RealSense conectada.")

    for dev in devices:
        try:
            name = dev.get_info(rs.camera_info.name)
            product_line = dev.get_info(rs.camera_info.product_line)
            print(f"Dispositivo encontrado: {name} | línea: {product_line}")
            if product_line == "D400":
                return dev
        except Exception:
            continue

    raise RuntimeError("No se encontró un dispositivo de la serie D400.")


def load_viewer_settings(settings_file):
    device = find_d400_device()
    adv = rs.rs400_advanced_mode(device)

    if not adv.is_enabled():
        print("Activando advanced mode...")
        adv.toggle_advanced_mode(True)
        time.sleep(2.0)

        # Puede reconectarse
        device = find_d400_device()
        adv = rs.rs400_advanced_mode(device)

    with open(settings_file, "r", encoding="utf-8") as f:
        json_text = f.read()

    print(f"Cargando settings desde: {settings_file}")
    adv.load_json(json_text)
    time.sleep(1.0)
    print("Settings del viewer aplicados correctamente.")


def apply_post_processing(depth_frame,
                          decimation,
                          depth_to_disparity,
                          spatial,
                          temporal,
                          disparity_to_depth,
                          hole_filling,
                          decimation_enabled=True,
                          spatial_enabled=True,
                          temporal_enabled=True,
                          hole_filling_enabled=True):
    frame = depth_frame

    if decimation_enabled:
        frame = decimation.process(frame)

    frame = depth_to_disparity.process(frame)

    if spatial_enabled:
        frame = spatial.process(frame)

    if temporal_enabled:
        frame = temporal.process(frame)

    frame = disparity_to_depth.process(frame)

    if hole_filling_enabled:
        frame = hole_filling.process(frame)

    return frame


def depth_to_gray(depth_image, depth_scale, min_m=0.4, max_m=2.5):
    depth_m = depth_image.astype(np.float32) * depth_scale
    depth_clipped = np.clip(depth_m, min_m, max_m)

    # Evitar división por cero
    denom = max(max_m - min_m, 1e-6)
    depth_gray = ((depth_clipped - min_m) / denom * 255).astype(np.uint8)
    return depth_gray


def main():
    save_dir = "captures_realsense"
    os.makedirs(save_dir, exist_ok=True)

    # 1) Cargar JSON exportado del viewer
    if os.path.exists(SETTINGS_FILE):
        try:
            load_viewer_settings(SETTINGS_FILE)
        except Exception as e:
            print(f"No se pudieron cargar los settings JSON: {e}")
            print("Se continuará con configuración manual.")
    else:
        print(f"No se encontró el archivo {SETTINGS_FILE}. Se continuará sin cargar JSON.")

    # 2) Pipeline Opción A: ambas a 848x480
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    align = rs.align(rs.stream.color)

    depth_scale = depth_sensor.get_depth_scale()
    print("Depth scale:", depth_scale)

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

    # Respaldo por si el JSON no cargó
    safe_set(depth_sensor, rs.option.emitter_enabled, 1)

    if autoexp_info:
        current_autoexp = safe_get(depth_sensor, rs.option.enable_auto_exposure, None)
        if current_autoexp is None:
            safe_set(depth_sensor, rs.option.enable_auto_exposure, 1)

    if laser_info:
        laser_min, laser_max, laser_step, _ = laser_info
        current_laser = safe_get(depth_sensor, rs.option.laser_power, None)
        if current_laser is None:
            safe_set(depth_sensor, rs.option.laser_power, laser_max)

    # 3) Filtros estilo viewer
    decimation = rs.decimation_filter()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # Ajustes conservadores
    decimation.set_option(rs.option.filter_magnitude, 2)

    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)

    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    decimation_enabled = True
    spatial_enabled = True
    temporal_enabled = True
    hole_filling_enabled = True

    min_m = 0.4
    max_m = 2.5

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
    print("  d = alternar decimation")
    print("  s = alternar spatial")
    print("  t = alternar temporal")
    print("  o = alternar hole filling")
    print("  z/x = bajar/subir min_m")
    print("  n/m = bajar/subir max_m")
    print("  c = capturar")
    print("  ESC = salir\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # IMPORTANTE:
            # Hacemos align primero para obtener depth y color ya registrados nativamente
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Luego aplicamos postproceso al depth ya alineado
            processed_depth_frame = apply_post_processing(
                depth_frame,
                decimation=decimation,
                depth_to_disparity=depth_to_disparity,
                spatial=spatial,
                temporal=temporal,
                disparity_to_depth=disparity_to_depth,
                hole_filling=hole_filling,
                decimation_enabled=decimation_enabled,
                spatial_enabled=spatial_enabled,
                temporal_enabled=temporal_enabled,
                hole_filling_enabled=hole_filling_enabled
            )

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(processed_depth_frame.get_data())

            # Si decimation cambia el tamaño, lo reescalamos al color
            if depth_image.shape[:2] != color_image.shape[:2]:
                depth_image = cv2.resize(
                    depth_image,
                    (color_image.shape[1], color_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            depth_gray = depth_to_gray(depth_image, depth_scale, min_m=min_m, max_m=max_m)
            depth_gray_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)

            overlay = cv2.addWeighted(color_image, 0.7, depth_gray_bgr, 0.3, 0)

            laser_val = safe_get(depth_sensor, rs.option.laser_power, -1)
            gain_val = safe_get(depth_sensor, rs.option.gain, -1)
            exposure_val = safe_get(depth_sensor, rs.option.exposure, -1)
            autoexp_val = safe_get(depth_sensor, rs.option.enable_auto_exposure, -1)

            status = (
                f"Laser:{laser_val:.1f}  "
                f"Gain:{gain_val:.1f}  "
                f"Exposure:{exposure_val:.1f}  "
                f"AutoExp:{int(autoexp_val) if autoexp_val is not None else -1}  "
                f"Dec:{'ON' if decimation_enabled else 'OFF'}  "
                f"Sp:{'ON' if spatial_enabled else 'OFF'}  "
                f"Tmp:{'ON' if temporal_enabled else 'OFF'}  "
                f"Hole:{'ON' if hole_filling_enabled else 'OFF'}  "
                f"Rango:[{min_m:.2f},{max_m:.2f}]m"
            )

            overlay_text = overlay.copy()
            cv2.putText(
                overlay_text,
                status,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.imshow("Color 848x480", color_image)
            cv2.imshow("Depth aligned 848x480 (Grayscale)", depth_gray)
            cv2.imshow("Overlay 848x480", overlay_text)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
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

            elif key == ord('d'):
                decimation_enabled = not decimation_enabled
                print(f"Decimation -> {'ON' if decimation_enabled else 'OFF'}")

            elif key == ord('s'):
                spatial_enabled = not spatial_enabled
                print(f"Spatial -> {'ON' if spatial_enabled else 'OFF'}")

            elif key == ord('t'):
                temporal_enabled = not temporal_enabled
                print(f"Temporal -> {'ON' if temporal_enabled else 'OFF'}")

            elif key == ord('o'):
                hole_filling_enabled = not hole_filling_enabled
                print(f"Hole filling -> {'ON' if hole_filling_enabled else 'OFF'}")

            elif key == ord('z'):
                min_m = max(0.1, min_m - 0.05)
                if min_m >= max_m:
                    min_m = max_m - 0.05
                print(f"min_m -> {min_m:.2f}")

            elif key == ord('x'):
                min_m = min(max_m - 0.05, min_m + 0.05)
                print(f"min_m -> {min_m:.2f}")

            elif key == ord('n'):
                max_m = max(min_m + 0.05, max_m - 0.05)
                print(f"max_m -> {max_m:.2f}")

            elif key == ord('m'):
                max_m = min(10.0, max_m + 0.05)
                print(f"max_m -> {max_m:.2f}")

            elif key == ord('c'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                color_path = os.path.join(save_dir, f"color_{timestamp}.png")
                depth_gray_path = os.path.join(save_dir, f"depth_gray_{timestamp}.png")
                overlay_path = os.path.join(save_dir, f"overlay_{timestamp}.png")
                depth_raw_path = os.path.join(save_dir, f"depth_raw_{timestamp}.npy")

                cv2.imwrite(color_path, color_image)
                cv2.imwrite(depth_gray_path, depth_gray)
                cv2.imwrite(overlay_path, overlay_text)
                np.save(depth_raw_path, depth_image)

                print("Captura guardada:")
                print(f"  {color_path}")
                print(f"  {depth_gray_path}")
                print(f"  {overlay_path}")
                print(f"  {depth_raw_path}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()