# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# region : File Description and Imports

"""
qcar_yolo_control.py

Combined script: Runs QCar speed & steering control in a background thread
while running two YOLOv8 object segmentation models concurrently in the main thread.
"""
import os
import signal
import numpy as np
from threading import Thread
import time
import cv2

# --- YOLO Vision Imports ---
from pit.YOLO.nets import YOLOv8
from pit.YOLO.utils import QCar2DepthAligned

# --- QCar Control Imports ---
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap

# ================ Experiment Configuration ================
# ===== Vehicle Control Timing Parameters
tf = 6000
startDelay = 1
controllerUpdateRate = 100

# ===== Vision Timing Parameters
sampleRate = 30.0
sampleTime = 1 / sampleRate
imageWidth = 640
imageHeight = 480

# ===== Speed Controller Parameters
v_ref = 0.5
K_p = 0.1
K_i = 1

# ===== Steering Controller Parameters
enableSteeringControl = True
K_stanley = 1
nodeSequence = [10, 4, 20, 10]

# endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# region : Initial setup
if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
else:
    initialPose = [0, 0, 0]

if not IS_PHYSICAL_QCAR:
    import qlabs_setup

    qlabs_setup.setup(
        initialPosition=[initialPose[0], initialPose[1], 0],
        initialOrientation=[0, 0, initialPose[2]],
    )
    calibrate = False
else:
    calibrate = "y" in input("do you want to recalibrate?(y/n)")

calibrationPose = [0, 2, -np.pi / 2]

# Used to enable safe keyboard triggered shutdown
global KILL_THREAD
KILL_THREAD = False


def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True


signal.signal(signal.SIGINT, sig_handler)
# endregion


# region : Controllers
class SpeedController:
    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3
        self.kp = kp
        self.ki = ki
        self.ei = 0

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        return np.clip(
            self.kp * e + self.ki * self.ei, -self.maxThrottle, self.maxThrottle
        )


class SteeringController:
    def __init__(self, waypoints, k=1, cyclic=True):
        self.maxSteeringAngle = np.pi / 6
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic
        self.p_ref = (0, 0)
        self.th_ref = 0

    def update(self, p, th, speed):
        wp_1 = self.wp[:, np.mod(self.wpi, self.N - 1)]
        wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]

        v = wp_2 - wp_1
        v_mag = np.linalg.norm(v)
        try:
            v_uv = v / v_mag
        except ZeroDivisionError:
            return 0

        tangent = np.arctan2(v_uv[1], v_uv[0])
        s = np.dot(p - wp_1, v_uv)

        if s >= v_mag:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1

        ep = wp_1 + v_uv * s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        ect = np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent - th)

        self.p_ref = ep
        self.th_ref = tangent

        return np.clip(
            wrap_to_pi(psi + np.arctan2(self.k * ect, speed)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle,
        )


# endregion


# region : Vehicle Control Loop (Runs in Background Thread)
def controlLoop():
    global KILL_THREAD
    u = 0
    delta = 0

    speedController = SpeedController(kp=K_p, ki=K_i)
    if enableSteeringControl:
        steeringController = SteeringController(waypoints=waypointSequence, k=K_stanley)

    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    if enableSteeringControl:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)
    else:
        gps = memoryview(b"")

    with qcar, gps:
        t0 = time.time()
        t = 0
        while (t < tf + startDelay) and (not KILL_THREAD):
            tp = t
            t = time.time() - t0
            dt = t - tp

            qcar.read()
            if enableSteeringControl:
                if gps.readGPS():
                    y_gps = np.array(
                        [gps.position[0], gps.position[1], gps.orientation[2]]
                    )
                    ekf.update([qcar.motorTach, delta], dt, y_gps, qcar.gyroscope[2])
                else:
                    ekf.update([qcar.motorTach, delta], dt, None, qcar.gyroscope[2])

                x = ekf.x_hat[0, 0]
                y = ekf.x_hat[1, 0]
                th = ekf.x_hat[2, 0]
                p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2
            v = qcar.motorTach

            if t < startDelay:
                u = 0
                delta = 0
            else:
                u = speedController.update(v, v_ref, dt)
                if enableSteeringControl:
                    delta = steeringController.update(p, th, v)
                else:
                    delta = 0

            qcar.write(u, delta)

        qcar.read_write_std(throttle=0, steering=0)


# endregion
def is_orange(image, box_xyxy):
    """
    Checks if a bounding box area contains the specific crimson-red
    (RGB: 200, 61, 79) of the lab traffic cones.
    """
    x1, y1, x2, y2 = map(int, box_xyxy)

    # Bound coordinates to image size to prevent crashes
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Extract the Region of Interest (ROI)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    # Convert to HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # --- HYPER-SPECIFIC CRIMSON BOUNDS ---
    # The target RGB translates to an HSV of approx [172, 177, 200].
    # We build a tight window around this exact value to avoid other red objects.
    lower_crimson = np.array([165, 130, 150])
    upper_crimson = np.array([179, 220, 255])

    mask = cv2.inRange(hsv_roi, lower_crimson, upper_crimson)

    # Calculate ratio of target pixels vs total pixels in the box
    target_pixels = cv2.countNonZero(mask)
    total_pixels = roi.shape[0] * roi.shape[1]

    # Require at least 15% of the bounding box to be this exact crimson
    return (total_pixels > 0) and ((target_pixels / total_pixels) > 0.15)


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# region : Setup and Run Experiment (Main Thread)
if __name__ == "__main__":

    # 1. Initialize YOLOv8 Models and Camera Stream FIRST
    print("Loading YOLO models and warming up TensorRT... please wait.")

    # Load Model 1 (Traffic Cones)
    cone_yolo = YOLOv8(
        modelPath="traffic_cones.pt",
        imageHeight=imageHeight,
        imageWidth=imageWidth,
    )

    # Load Model 2 (General Segmentation)
    seg_yolo = YOLOv8(
        modelPath="yolov8s-seg.pt",
        imageHeight=imageHeight,
        imageWidth=imageWidth,
    )

    QCarImg = QCar2DepthAligned()

    # 2. START VEHICLE CONTROL THREAD SECOND
    print("Starting QCar hardware and GPS connection...")
    controlThread = Thread(target=controlLoop)
    controlThread.start()

    print("Experiment started. Press Ctrl+C in terminal to stop.")
    try:
        # Main Vision Loop
        while controlThread.is_alive() and (not KILL_THREAD):
            start = time.time()

            # Read Images
            QCarImg.read()

            # --- Process with Model 1 (Cones) ---
            # --- Process with Model 1 (Cones) ---
            # --- Process with Model 1 (Cones) ---
            # --- Process with Model 1 (Cones) ---
            rgbProcessed_cones = cone_yolo.pre_process(QCarImg.rgb)
            prediction_cones = cone_yolo.predict(
                inputImg=rgbProcessed_cones,
                classes=[0],
                confidence=0.05,
                half=True,
                verbose=False,
            )

            # Safely grab the raw YOLO results object
            res = (
                prediction_cones[0]
                if isinstance(prediction_cones, list)
                else prediction_cones
            )

            # Let Quanser calculate the depth for all detections first
            raw_results_cones = cone_yolo.post_processing(
                alignedDepth=QCarImg.depth, clippingDistance=5
            )

            keep_indices = []
            filtered_res = None
            results_cones = []

            # 1. Identify which bounding boxes are actually orange
            if res is not None and res.boxes is not None and len(res.boxes) > 0:
                for i, box in enumerate(res.boxes.xyxy.cpu().numpy()):
                    if is_orange(rgbProcessed_cones, box):
                        keep_indices.append(i)

                # 2. Slice the YOLO results to keep only orange cones for drawing
                filtered_res = res[keep_indices]

                # 3. Filter the Quanser data objects for the terminal flood
                for i in keep_indices:
                    if i < len(raw_results_cones):
                        results_cones.append(raw_results_cones[i])

            # --- Process with Model 2 (General Seg) ---
            rgbProcessed_seg = seg_yolo.pre_process(QCarImg.rgb)
            prediction_seg = seg_yolo.predict(
                inputImg=rgbProcessed_seg,
                classes=[2, 9, 11, 33],
                confidence=0.3,
                half=True,
                verbose=False,
            )
            results_seg = seg_yolo.post_processing(
                alignedDepth=QCarImg.depth, clippingDistance=5
            )

            # --- Combine Results ---
            all_detected_objects = results_cones + results_seg

            # Print Detections (Logging objects from both models)
            for obj in all_detected_objects:
                print(obj.__dict__)

            # --- High-Speed Rendering ---
            # 1. Generate the base annotated image from the general segmentation model
            # We leave showFPS=True here so Quanser handles the framerate text
            combinedImg = seg_yolo.post_process_render(showFPS=True)

            # 2. Draw the verified orange cones directly on top using Ultralytics
            # This completely bypasses the Quanser tensor error and avoids image dimming!
            if filtered_res is not None and len(filtered_res) > 0:
                combinedImg = filtered_res.plot(img=combinedImg)

            # 3. Show the result
            cv2.imshow("Object Segmentation", combinedImg)

            # Timing & Sleep logic
            end = time.time()
            computationTime = end - start
            sleepTime = sampleTime - (computationTime % sampleTime)

            msSleepTime = int(1000 * sleepTime)
            if msSleepTime <= 0:
                msSleepTime = 1

            # WaitKey handles the image updating and loop delay
            cv2.waitKey(msSleepTime)

    except KeyboardInterrupt:
        print("\n[INFO] User interrupted! Shutting down...")
    finally:
        # Trigger thread kill flag just in case
        KILL_THREAD = True

        # Clean up vision resources
        QCarImg.terminate()
        cv2.destroyAllWindows()

        # Clean up QLabs if virtual
        if not IS_PHYSICAL_QCAR:
            import qlabs_setup

            qlabs_setup.terminate()

        # Wait for the control thread to finish safely
        if controlThread.is_alive():
            controlThread.join(timeout=2)

        print("Experiment complete.")
# endregion
