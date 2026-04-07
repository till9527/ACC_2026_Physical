#!/usr/bin/env python3
import time
import numpy as np

# --- Quanser Hardware & Math Imports ---
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from hal.content.qcar_functions import QCarEKF
from pal.utilities.math import wrap_to_pi

# --- Custom Map Import ---
from custom_roadmap import CustomRoadMap

# =========================================================
# CONFIGURATION
# =========================================================
BASE_SPEED = 0.5  # m/s (Start slow on the real car!)
MAX_STEERING = np.pi / 6  # 30 degrees

# Replace with the nodes you want to test from custom_roadmap.py
NODE_SEQUENCE = [0, 2, 4, 14, 20, 22, 9, 0]

# Controller Tuning
K_STANLEY = 1.0
K_P_SPEED = 0.1
K_I_SPEED = 1.0


# =========================================================
# CONTROLLERS
# =========================================================
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
    def __init__(self, waypoints, k=1, cyclic=False):
        self.maxSteeringAngle = MAX_STEERING
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic

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

        # Move to next waypoint if we pass the current segment
        if s >= v_mag:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1

        ep = wp_1 + v_uv * s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        ect = np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent - th)

        # Stanley Control Law
        return np.clip(
            wrap_to_pi(psi + np.arctan2(self.k * ect, speed + 0.001)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle,
        )


# =========================================================
# MAIN LOOP
# =========================================================
if __name__ == "__main__":

    if not IS_PHYSICAL_QCAR:
        print("[ERROR] This script is meant for the physical QCar!")
        exit()

    # 1. Setup Roadmap
    print("Generating Path from CustomRoadMap...")
    roadmap = CustomRoadMap()
    waypointSequence = roadmap.generate_path(NODE_SEQUENCE)

    theoretical_start = roadmap.get_node_pose(NODE_SEQUENCE[0]).squeeze()

    # 2. Initialize GPS to find real-world location
    print("Connecting to GPS to measure actual offset...")
    gps = QCarGPS(initialPose=theoretical_start, calibrate=True)
    actual_pose = None

    with gps:
        # Take 50 quick samples to ensure we have a stable lock
        for _ in range(50):
            if gps.readGPS():
                actual_pose = np.array(
                    [gps.position[0], gps.position[1], gps.orientation[2]]
                )
            time.sleep(0.01)

    if actual_pose is None:
        print(
            "[WARNING] Could not get GPS fix. Falling back to theoretical coordinates."
        )
        actual_pose = theoretical_start
    else:
        print(f"-> Theoretical Node 0: {theoretical_start}")
        print(f"-> Actual GPS Pose:    {actual_pose}")

    # 3. Dynamic Coordinate Calibration (Affine Transformation)
    print("Calibrating waypoint map to actual position...")
    dtheta = actual_pose[2] - theoretical_start[2]

    # Shift to theoretical origin
    shifted_x = waypointSequence[0, :] - theoretical_start[0]
    shifted_y = waypointSequence[1, :] - theoretical_start[1]

    # Apply Rotation
    cos_t = np.cos(dtheta)
    sin_t = np.sin(dtheta)
    rotated_x = shifted_x * cos_t - shifted_y * sin_t
    rotated_y = shifted_x * sin_t + shifted_y * cos_t

    # Translate to Actual Origin
    waypointSequence[0, :] = rotated_x + actual_pose[0]
    waypointSequence[1, :] = rotated_y + actual_pose[1]

    # 4. Setup Controllers and EKF
    speedController = SpeedController(kp=K_P_SPEED, ki=K_I_SPEED)
    steeringController = SteeringController(
        waypoints=waypointSequence, k=K_STANLEY, cyclic=False
    )

    qcar = QCar(readMode=1, frequency=100)
    # Start the EKF at our calibrated ACTUAL pose
    ekf = QCarEKF(x_0=actual_pose)

    print("--- STARTING HARDWARE RUN ---")
    print("Press Ctrl+C to stop.")

    with qcar, gps:
        t0 = time.time()
        t = 0

        # Initialize our command variables
        throttle = 0.0
        steering = 0.0

        try:
            while True:
                # Timing
                tp = t
                t = time.time() - t0
                dt = t - tp

                # Read Sensors
                qcar.read()

                # Update EKF
                if gps.readGPS():
                    y_gps = np.array(
                        [gps.position[0], gps.position[1], gps.orientation[2]]
                    )
                    ekf.update([qcar.motorTach, steering], dt, y_gps, qcar.gyroscope[2])
                else:
                    ekf.update([qcar.motorTach, steering], dt, None, qcar.gyroscope[2])

                # Current State
                x = ekf.x_hat[0, 0]
                y = ekf.x_hat[1, 0]
                th = ekf.x_hat[2, 0]
                current_speed = qcar.motorTach

                # Calculate kinematics from front axle
                p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2

                # Calculate Control Outputs
                throttle = speedController.update(current_speed, BASE_SPEED, dt)
                steering = steeringController.update(p, th, current_speed)

                # Write to Motors
                qcar.write(throttle, steering)

                # Stop if we reached the final waypoint segment
                if steeringController.wpi >= len(waypointSequence[0, :]) - 2:
                    print("[INFO] Reached the end of the waypoint sequence!")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Manual Override! Shutting down...")
        finally:
            print("[INFO] Stopping motors.")
            qcar.read_write_std(throttle=0, steering=0)
