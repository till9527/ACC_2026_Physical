"""
qcar_yolo_visualized.py

Combined script: runs QCar speed and steering control in a background thread
while running two YOLOv8 object segmentation models concurrently in the main
thread. Now includes real-time PyQtGraph MultiScope visualization for telemetry
and roadmap tracking.
"""

import os
import signal
import threading
import time
from threading import Lock, Thread

import cv2
import numpy as np
import pyqtgraph as pg

# --- YOLO Vision Imports ---
from pit.YOLO.nets import YOLOv8
from pit.YOLO.utils import QCar2DepthAligned

# --- QCar Control & Visualization Imports ---
from hal.content.qcar_functions import QCarEKF
from pal.utilities.vision import Camera2D
from pal.products.qcar import IS_PHYSICAL_QCAR, QCar, QCarGPS
from pal.utilities.math import wrap_to_pi
from pal.utilities.scope import MultiScope
import pal.resources.images as images

# --- Roadmap Imports ---
from custom_roadmap import CustomRoadMap

# ============================
# Experiment Configuration
# ============================
CAMERA_ID = "2"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FRAME_RATE = 30
# Vehicle control timing parameters
TF = 6000
START_DELAY = 1.0
CONTROLLER_UPDATE_RATE = 100

# Vision timing parameters
SAMPLE_RATE = 30.0
SAMPLE_TIME = 1.0 / SAMPLE_RATE
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Speed controller parameters
V_REF = 0.5
K_P = 0.1
K_I = 1.0
frame_lock = threading.Lock()

# Steering controller parameters
ENABLE_STEERING_CONTROL = True
K_STANLEY = 1.0
NODE_SEQUENCE = [10, 4, 20, 10]

# Object-handling parameters
STOP_SIGN_MIN_WIDTH = 50
YIELD_SIGN_MIN_WIDTH = 60
STOP_SIGN_WAIT_TIME_S = 3.0
YIELD_SIGN_WAIT_TIME_S = 0.5
SIGN_REARM_TIME_S = 0.25

CAMERA_FRAME_WIDTH = 640
CONE_CENTER_THIRD_MIN_X = CAMERA_FRAME_WIDTH / 3.0
CONE_CENTER_THIRD_MAX_X = 2.0 * CAMERA_FRAME_WIDTH / 3.0
CONE_MIN_WIDTH = 20
CONE_MIN_HEIGHT = 20
CONE_AVOID_LEFT_STEER = np.deg2rad(18.0)
CONE_AVOID_RIGHT_STEER = np.deg2rad(-18.0)
CONE_AVOID_PHASE_DURATION_S = 1.5
CONE_AVOID_COOLDOWN_S = 1.0
CONE_AVOID_TARGET_SPEED = 0.3

DEBUG_DETECTIONS = False


# ============================
# Initial setup
# ============================
def check_map_health(roadmap):
    """Scans every edge in the roadmap and reports any failures."""
    print("\n=== EDGE HEALTH REPORT ===")
    failed_count = 0

    for edge in roadmap.edges:
        if (
            getattr(edge, "waypoints", None) is None
            or len(edge.waypoints) == 0
            or len(edge.waypoints[0]) == 0
        ):
            from_id = roadmap.nodes.index(edge.fromNode)
            to_id = roadmap.nodes.index(edge.toNode)
            print(
                f" [FAILED] Edge {from_id} -> {to_id} could not generate geometry! Check radii and node coordinates to avoid curb crossover."
            )
            failed_count += 1

    if failed_count == 0:
        print(" [OK] All edges generated perfectly!")
    else:
        print(f" >>> WARNING: {failed_count} edges are broken on the map! <<<")
    print("==========================\n")


if ENABLE_STEERING_CONTROL:
    custom_roadmap_joe = CustomRoadMap()
    check_map_health(custom_roadmap_joe)
    waypoint_sequence = custom_roadmap_joe.generate_path(NODE_SEQUENCE)

    if waypoint_sequence is None:
        print("\n[ERROR] Path generation failed! Tracing NODE_SEQUENCE for errors...")
        import sys

        sys.exit(1)

    initial_pose = custom_roadmap_joe.get_node_pose(NODE_SEQUENCE[0]).squeeze()
else:
    initial_pose = [0, 0, 0]

if not IS_PHYSICAL_QCAR:
    import qlabs_setup

    qlabs_setup.setup(
        initialPosition=[initial_pose[0], initial_pose[1], 0],
        initialOrientation=[0, 0, initial_pose[2]],
    )
    calibrate = False
else:
    calibrate = "y" in input("Do you want to recalibrate? (y/n): ").lower()

calibration_pose = [0, 2, -np.pi / 2]

KILL_THREAD = False
latest_detections = []
detections_lock = Lock()


def sig_handler(*_args):
    global KILL_THREAD
    KILL_THREAD = True


signal.signal(signal.SIGINT, sig_handler)


# ============================
# Controllers
# ============================
class SpeedController:
    def __init__(self, kp=0.0, ki=0.0, max_throttle=0.3, integral_limit=1.0):
        self.max_throttle = max_throttle
        self.kp = kp
        self.ki = ki
        self.ei = 0.0
        self.integral_limit = integral_limit

    def reset(self):
        self.ei = 0.0

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        self.ei = float(np.clip(self.ei, -self.integral_limit, self.integral_limit))
        u = self.kp * e + self.ki * self.ei
        return float(np.clip(u, -self.max_throttle, self.max_throttle))


class SteeringController:
    def __init__(self, waypoints, k=1.0, cyclic=True):
        self.max_steering_angle = np.pi / 6
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic
        self.p_ref = (0, 0)
        self.th_ref = 0
        self.speed_epsilon = 1e-3

    def update(self, p, th, speed):
        wp_1 = self.wp[:, np.mod(self.wpi, self.N - 1)]
        wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]

        v = wp_2 - wp_1
        v_mag = np.linalg.norm(v)
        if v_mag < 1e-9:
            return 0.0

        v_uv = v / v_mag
        tangent = np.arctan2(v_uv[1], v_uv[0])

        s = np.dot(p - wp_1, v_uv)
        dist_to_next = np.linalg.norm(p - wp_2)

        if s >= v_mag or dist_to_next < 0.1:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1

        ep = wp_1 + v_uv * s
        ct = ep - p
        heading_to_path = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        ect = np.linalg.norm(ct) * np.sign(heading_to_path)
        psi = wrap_to_pi(tangent - th)
        speed_for_control = max(abs(speed), self.speed_epsilon)

        self.p_ref = ep
        self.th_ref = tangent

        delta = wrap_to_pi(psi + np.arctan2(self.k * ect, speed_for_control))
        return float(np.clip(delta, -self.max_steering_angle, self.max_steering_angle))


# ============================
# Detection helpers
# ============================
def normalize_class_name(name):
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def extract_detection_metadata(result):
    detections = []
    if result is None:
        return detections

    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "xywh", None) is None:
        return detections

    names = getattr(result, "names", {}) or {}
    xywh = boxes.xywh.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else []

    for box, class_id in zip(xywh, cls):
        x_center, y_center, width, height = box.tolist()
        detections.append(
            {
                "class": names.get(class_id, str(class_id)),
                "x": float(x_center),
                "y": float(y_center),
                "width": float(width),
                "height": float(height),
            }
        )
    return detections


def extract_quanser_detection_metadata(results):
    detections = []
    if results is None:
        return detections
    for obj in results:
        name = str(getattr(obj, "name", "")).strip()
        detection = {
            "class": name,
            "x": float(getattr(obj, "x", 0.0)),
            "y": float(getattr(obj, "y", 0.0)),
            "width": float(getattr(obj, "width", 0.0)),
            "height": float(getattr(obj, "height", 0.0)),
        }
        light_color = getattr(obj, "lightColor", None)
        if light_color is not None and "traffic light" in name.lower():
            color = str(light_color).strip().lower()
            if color:
                detection["class"] = f"traffic light ({color} )"
        detections.append(detection)
    return detections


def extract_sign_detection_metadata(result):
    detections = []
    for det in extract_detection_metadata(result):
        cls = normalize_class_name(det["class"])
        if cls in ("stop sign", "yield sign"):
            det["class"] = cls
            detections.append(det)
    return detections


def is_orange(image, box_xyxy):
    x1, y1, x2, y2 = map(int, box_xyxy)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_crimson = np.array([165, 130, 150])
    upper_crimson = np.array([179, 220, 255])
    mask = cv2.inRange(hsv_roi, lower_crimson, upper_crimson)

    target_pixels = cv2.countNonZero(mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    return (total_pixels > 0) and ((target_pixels / total_pixels) > 0.15)


# ============================
# Vehicle Control Loop
# ============================
def control_loop():
    global KILL_THREAD
    global latest_detections

    u = 0.0
    delta = 0.0

    countMax = CONTROLLER_UPDATE_RATE / 10
    count = 0

    speed_controller = SpeedController(kp=K_P, ki=K_I)
    steering_controller = (
        SteeringController(waypoints=waypoint_sequence, k=K_STANLEY)
        if ENABLE_STEERING_CONTROL
        else None
    )

    stop_sign_start_time = 0.0
    yield_sign_start_time = 0.0
    stop_sign_last_seen_time = 0.0
    yield_sign_last_seen_time = 0.0
    is_stopped_for_sign = False
    is_stopped_for_yield = False
    cone_avoid_active = False
    cone_avoid_start_time = 0.0
    cone_avoid_cooldown_until = 0.0

    qcar = QCar(readMode=1, frequency=CONTROLLER_UPDATE_RATE)
    if ENABLE_STEERING_CONTROL:
        ekf = QCarEKF(x_0=initial_pose)
        gps = QCarGPS(initialPose=calibration_pose, calibrate=calibrate)
    else:
        ekf = None
        gps = memoryview(b"")

    with qcar, gps:
        t0 = time.time()
        t = 0.0

        while (t < TF + START_DELAY) and (not KILL_THREAD):
            tp = t
            t = time.time() - t0
            dt = max(t - tp, 1e-3)
            current_time = time.time()

            qcar.read()
            v = float(qcar.motorTach)

            if ENABLE_STEERING_CONTROL:
                if gps.readGPS():
                    y_gps = np.array(
                        [gps.position[0], gps.position[1], gps.orientation[2]]
                    )
                    ekf.update([qcar.motorTach, delta], dt, y_gps, qcar.gyroscope[2])
                else:
                    ekf.update([qcar.motorTach, delta], dt, None, qcar.gyroscope[2])

                x = float(ekf.x_hat[0, 0])
                y = float(ekf.x_hat[1, 0])
                th = float(ekf.x_hat[2, 0])
                p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2
            else:
                th = 0.0
                p = np.array([0.0, 0.0])

            with detections_lock:
                current_detections = list(latest_detections)

            stop_sign_visible = False
            yield_sign_visible = False
            red_or_yellow_light_visible = False
            green_light_visible = False
            cone_detected_center = False

            for obj in current_detections:
                obj_class = normalize_class_name(obj["class"])
                if (
                    obj_class in ("stop sign", "stopsign")
                    and obj["width"] > STOP_SIGN_MIN_WIDTH
                ):
                    stop_sign_visible = True
                    stop_sign_last_seen_time = current_time
                elif (
                    obj_class in ("yield sign", "yieldsign")
                    and obj["width"] > YIELD_SIGN_MIN_WIDTH
                ):
                    yield_sign_visible = True
                    yield_sign_last_seen_time = current_time
                elif obj_class in ("traffic light (green )", "traffic light (green)"):
                    green_light_visible = True
                elif obj_class in (
                    "traffic light (red )",
                    "traffic light (red)",
                    "traffic light (yellow )",
                    "traffic light (yellow)",
                ):
                    red_or_yellow_light_visible = True
                elif obj_class in ("traffic cone", "traffic-cone"):
                    if (
                        CONE_CENTER_THIRD_MIN_X < obj["x"] < CONE_CENTER_THIRD_MAX_X
                        and obj["height"] > CONE_MIN_HEIGHT
                        and obj["width"] > CONE_MIN_WIDTH
                    ):
                        cone_detected_center = True

            should_stop = red_or_yellow_light_visible
            if green_light_visible and not red_or_yellow_light_visible:
                should_stop = False

            if yield_sign_visible and not is_stopped_for_yield:
                should_stop = True
                if yield_sign_start_time == 0.0:
                    yield_sign_start_time = current_time
                elif current_time - yield_sign_start_time > YIELD_SIGN_WAIT_TIME_S:
                    is_stopped_for_yield = True
                    yield_sign_start_time = 0.0
            elif (
                not yield_sign_visible
                and current_time - yield_sign_last_seen_time > SIGN_REARM_TIME_S
            ):
                yield_sign_start_time = 0.0
                is_stopped_for_yield = False

            if stop_sign_visible and not is_stopped_for_sign:
                should_stop = True
                if stop_sign_start_time == 0.0:
                    stop_sign_start_time = current_time
                elif current_time - stop_sign_start_time > STOP_SIGN_WAIT_TIME_S:
                    is_stopped_for_sign = True
                    stop_sign_start_time = 0.0
            elif (
                not stop_sign_visible
                and current_time - stop_sign_last_seen_time > SIGN_REARM_TIME_S
            ):
                stop_sign_start_time = 0.0
                is_stopped_for_sign = False

            if (
                cone_detected_center
                and (not cone_avoid_active)
                and current_time >= cone_avoid_cooldown_until
            ):
                cone_avoid_active = True
                cone_avoid_start_time = current_time

            cone_avoid_steer = 0.0
            target_speed = V_REF
            if cone_avoid_active:
                should_stop = False
                target_speed = min(V_REF, CONE_AVOID_TARGET_SPEED)
                cone_avoid_elapsed = current_time - cone_avoid_start_time
                if cone_avoid_elapsed < CONE_AVOID_PHASE_DURATION_S:
                    cone_avoid_steer = CONE_AVOID_LEFT_STEER
                elif cone_avoid_elapsed < (2.0 * CONE_AVOID_PHASE_DURATION_S):
                    cone_avoid_steer = CONE_AVOID_RIGHT_STEER
                else:
                    cone_avoid_active = False
                    cone_avoid_cooldown_until = current_time + CONE_AVOID_COOLDOWN_S
                    cone_avoid_steer = 0.0
                    target_speed = V_REF

            if t < START_DELAY or should_stop:
                u = 0.0
                delta = 0.0
                speed_controller.reset()
            else:
                u = speed_controller.update(v, target_speed, dt)
                if ENABLE_STEERING_CONTROL and steering_controller is not None:
                    delta = (
                        cone_avoid_steer
                        if cone_avoid_active
                        else steering_controller.update(p, th, v)
                    )
                else:
                    delta = 0.0

            qcar.write(u, delta)

            # Update PyQt Scopes
            count += 1
            if count >= countMax and t > START_DELAY:
                t_plot = t - START_DELAY

                # Dynamic plotting of target speed vs actual
                # speedScope.axes[0].sample(t_plot, [v, target_speed])
                # speedScope.axes[1].sample(t_plot, [target_speed - v])
                # speedScope.axes[2].sample(t_plot, [u])

                if ENABLE_STEERING_CONTROL:
                    steeringScope.axes[0].sample(t_plot, [[p[0], p[1]]])

                    x_ref = (
                        steering_controller.p_ref[0] if steering_controller else p[0]
                    )
                    y_ref = (
                        steering_controller.p_ref[1] if steering_controller else p[1]
                    )
                    th_ref = steering_controller.th_ref if steering_controller else th

                    # steeringScope.axes[0].sample(t_plot, [p[0], x_ref])
                    # steeringScope.axes[1].sample(t_plot, [p[1], y_ref])
                    # steeringScope.axes[2].sample(t_plot, [th, th_ref])
                    # steeringScope.axes[3].sample(t_plot, [delta])

                    arrow.setPos(p[0], p[1])
                    arrow.setStyle(angle=180 - th * 180 / np.pi)

                count = 0

        qcar.read_write_std(throttle=0, steering=0)


# ============================
# Main experiment loop
# ============================
if __name__ == "__main__":

    # --- 1. SETUP SCOPES ---
    fps = 10 if IS_PHYSICAL_QCAR else 30

    # speedScope = MultiScope(rows=3, cols=1, title="Vehicle Speed Control", fps=fps)
    # speedScope.addAxis(row=0, col=0, timeWindow=TF, yLabel="Vehicle Speed [m/s]", yLim=(0, 1))
    # speedScope.axes[0].attachSignal(name="v_meas", width=2)
    # speedScope.axes[0].attachSignal(name="target_speed")

    # speedScope.addAxis(row=1, col=0, timeWindow=TF, yLabel="Speed Error [m/s]", yLim=(-0.5, 0.5))
    # speedScope.axes[1].attachSignal()

    # speedScope.addAxis(row=2, col=0, timeWindow=TF, xLabel="Time [s]", yLabel="Throttle Command [%]", yLim=(-0.3, 0.3))
    # speedScope.axes[2].attachSignal()

    if ENABLE_STEERING_CONTROL:
        steeringScope = MultiScope(
            rows=4, cols=2, title="Vehicle Steering Control", fps=fps
        )

        # steeringScope.addAxis(row=0, col=0, timeWindow=TF, yLabel="x Position [m]", yLim=(-2.5, 2.5))
        # steeringScope.axes[0].attachSignal(name="x_meas")
        # steeringScope.axes[0].attachSignal(name="x_ref")

        # steeringScope.addAxis(row=1, col=0, timeWindow=TF, yLabel="y Position [m]", yLim=(-1, 5))
        # steeringScope.axes[1].attachSignal(name="y_meas")
        # steeringScope.axes[1].attachSignal(name="y_ref")

        # steeringScope.addAxis(row=2, col=0, timeWindow=TF, yLabel="Heading Angle [rad]", yLim=(-3.5, 3.5))
        # steeringScope.axes[2].attachSignal(name="th_meas")
        # steeringScope.axes[2].attachSignal(name="th_ref")

        # steeringScope.addAxis(row=3, col=0, timeWindow=TF, yLabel="Steering Angle [rad]", yLim=(-0.6, 0.6))
        # steeringScope.axes[3].attachSignal()
        # steeringScope.axes[3].xLabel = "Time [s]"

        steeringScope.addXYAxis(
            row=0,
            col=1,
            rowSpan=4,
            xLabel="x Position [m]",
            yLabel="y Position [m]",
            xLim=(-2.5, 2.5),
            yLim=(-1, 5),
        )
        steeringScope.axes[0].plot.setAspectLocked(True)
        im = cv2.imread(images.SDCS_CITYSCAPE, cv2.IMREAD_GRAYSCALE)
        steeringScope.axes[0].attachImage(
            scale=(-0.002035, 0.002035),
            offset=(1125, 2365),
            rotation=180,
            levels=(0, 255),
        )
        steeringScope.axes[0].images[0].setImage(image=im)

        referencePath = pg.PlotDataItem(
            pen={"color": (85, 168, 104), "width": 2}, name="Reference"
        )
        steeringScope.axes[0].plot.addItem(referencePath)
        referencePath.setData(waypoint_sequence[0, :], waypoint_sequence[1, :])
        steeringScope.axes[0].attachSignal(name="Estimated", width=2)

        arrow = pg.ArrowItem(
            angle=180,
            tipAngle=60,
            headLen=10,
            tailLen=10,
            tailWidth=5,
            pen={"color": "w", "fillColor": [196, 78, 82], "width": 1},
            brush=[196, 78, 82],
        )
        arrow.setPos(initial_pose[0], initial_pose[1])
        steeringScope.axes[0].plot.addItem(arrow)

    # --- 2. START YOLO MODELS ---
    print("Loading YOLO models and warming up TensorRT... please wait.")
    cone_yolo = YOLOv8(
        modelPath="traffic_cones.pt", imageHeight=IMAGE_HEIGHT, imageWidth=IMAGE_WIDTH
    )
    seg_yolo = YOLOv8(
        modelPath="yolov8s-seg.pt", imageHeight=IMAGE_HEIGHT, imageWidth=IMAGE_WIDTH
    )
    qcar_img = QCar2DepthAligned()
    camera = Camera2D(
            cameraId=CAMERA_ID,
            frameWidth=IMAGE_WIDTH,
            frameHeight=IMAGE_HEIGHT,
            frameRate=FRAME_RATE,
        )
    # --- 3. START CONTROL THREAD ---
    print("Starting QCar hardware and GPS connection...")
    control_thread = Thread(target=control_loop, daemon=True)
    control_thread.start()
    threshold_value = 115 
    target_frame_time = 1.0 / FRAME_RATE
    # --- 4. RUN COMBINED MAIN LOOP ---
    print("Experiment started. Press Ctrl+C in terminal to stop.")
    try:
        while control_thread.is_alive() and (not KILL_THREAD):
            start = time.time()
            if camera.read():
                # 1. Get raw frame
                raw_frame = camera.imageData

                # 2. CROP TO LOWER HALF
                h, w = raw_frame.shape[:2]
                lower_half_frame = raw_frame[int(h / 2):h, :]

                # 3. CLEAN THRESHOLDING
                gray = cv2.cvtColor(lower_half_frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, binaryImage = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

                with frame_lock:
                    latest_frame = binaryImage

                # Display frame
                cv2.imshow("Processed Frame", binaryImage)
            # --- Object Detection Block ---
            qcar_img.read()
            rgb_processed_cones = cone_yolo.pre_process(qcar_img.rgb)
            prediction_cones = cone_yolo.predict(
                inputImg=rgb_processed_cones,
                classes=[0],
                confidence=0.05,
                half=True,
                verbose=False,
            )
            res_cones = (
                prediction_cones[0]
                if isinstance(prediction_cones, list)
                else prediction_cones
            )
            raw_results_cones = cone_yolo.post_processing(
                alignedDepth=qcar_img.depth, clippingDistance=5
            )

            keep_indices = []
            filtered_res_cones = None
            results_cones = []
            filtered_cone_metadata = []

            if (
                res_cones is not None
                and res_cones.boxes is not None
                and len(res_cones.boxes) > 0
            ):
                cone_metadata = extract_detection_metadata(res_cones)
                for i, box in enumerate(res_cones.boxes.xyxy.cpu().numpy()):
                    if is_orange(rgb_processed_cones, box):
                        keep_indices.append(i)
                        if i < len(cone_metadata):
                            filtered_cone_metadata.append(cone_metadata[i])

                if keep_indices:
                    filtered_res_cones = res_cones[keep_indices]
                    for i in keep_indices:
                        if i < len(raw_results_cones):
                            results_cones.append(raw_results_cones[i])

            rgb_processed_seg = seg_yolo.pre_process(qcar_img.rgb)
            prediction_seg = seg_yolo.predict(
                inputImg=rgb_processed_seg, confidence=0.3, half=True, verbose=False
            )
            res_seg = (
                prediction_seg[0]
                if isinstance(prediction_seg, list)
                else prediction_seg
            )
            results_seg = seg_yolo.post_processing(
                alignedDepth=qcar_img.depth, clippingDistance=5
            )

            sign_metadata = extract_sign_detection_metadata(res_seg)
            traffic_light_metadata = extract_quanser_detection_metadata(results_seg)
            combined_metadata = (
                filtered_cone_metadata + sign_metadata + traffic_light_metadata
            )

            with detections_lock:
                latest_detections.clear()
                latest_detections.extend(combined_metadata)

            combined_img = seg_yolo.post_process_render(showFPS=True)
            if filtered_res_cones is not None and len(filtered_res_cones) > 0:
                combined_img = filtered_res_cones.plot(img=combined_img)

            cv2.imshow("Object Segmentation", combined_img)

            # --- Qt Interface Refresh Block ---
            MultiScope.refreshAll()

            # --- Timing and Sleep Management ---
            end = time.time()
            computation_time = end - start
            sleep_time = max(SAMPLE_TIME - (computation_time % SAMPLE_TIME), 1e-3)
            ms_sleep_time = max(int(1000 * sleep_time), 1)

            cv2.waitKey(ms_sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] User interrupted! Shutting down...")
    finally:
        KILL_THREAD = True
        qcar_img.terminate()
        cv2.destroyAllWindows()

        if not IS_PHYSICAL_QCAR:
            import qlabs_setup

            qlabs_setup.terminate()

        if control_thread.is_alive():
            control_thread.join(timeout=2)

        print("Experiment complete.")
