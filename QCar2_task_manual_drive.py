## task_task_manual_drive.py
# This example demonstrates how to use the LogitechF710 to send throttle and steering
# commands to the QCar depending on 2 driving styles.
# Use the hardware_test_basic_io.py to troubleshoot uses trying to drive the QCar.

from pal.products.qcar import QCar
from pal.utilities.gamepad import LogitechF710
from pal.utilities.math import *

import os
import time
import struct
import numpy as np
import csv # Added for CSV recording

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Timing Parameters and methods
startTime = time.time()


def elapsed_time():
    return time.time() - startTime


sampleRate = 50
sampleTime = 1 / sampleRate
simulationTime = 60.0
print("Sample Time: ", sampleTime)

# Additional parameters
counter = 0

# Initialize motor command array
QCarCommand = np.array([0, 0])

# Set up a differentiator to get encoderSpeed from encoderCounts
diff = Calculus().differentiator_variable(sampleTime)
_ = next(diff)
timeStep = sampleTime

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## QCar and Gamepad Initialization
# Changing readmode to 0 to use imediate I/O
readMode = 0

myCar = QCar(readMode=readMode)
gpad = LogitechF710()

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Driving Configuration: Use 3 toggles or 4 toggles mode as you see fit:
# Common to both 3 or 4 mode
#   Steering                    - Left Lateral axis
#   Arm                         - buttonLeft
# In 3 mode:
#   Throttle (Drive or Reverse) - Right Longitudonal axis
# In 4 mode:
#   Throttle                    - Right Trigger (always positive)
#   Button A                    - Reverse if held, Drive otherwise
configuration = "4"  # change to '4' if required

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Reset startTime before Main Loop
startTime = time.time()

# CSV Recording Variables
is_recording = False
prev_x_state = False
csv_file = None
csv_writer = None

## Main Loop
try:
    while True:
        # Start timing this iteration
        start = elapsed_time()

        # Read Gamepad states
        new = gpad.read()
        
        # --- CSV Recording Toggle Logic ---
        if new:
            current_x_state = gpad.buttonX
            # Check for falling edge (button was pressed, now released)
            if prev_x_state and not current_x_state:
                is_recording = not is_recording # Toggle recording state
                
                if is_recording:
                    # Open file and write headers when recording starts
                    csv_file = open('steering_data.csv', 'w', newline='')
                    csv_writer = csv.writer(csv_file)
                    # Added Speed_mps to the headers
                    csv_writer.writerow(['Time_s', 'Steering_rad', 'Speed_mps'])
                else:
                    # Close file when recording stops
                    if csv_file:
                        csv_file.close()
                        csv_file = None
                        csv_writer = None
            
            prev_x_state = current_x_state
        # ----------------------------------

        # Basic IO - write motor commands
        if configuration == "3":
            if new and gpad.buttonLeft:
                QCarCommand = np.array(
                    [0.3 * gpad.rightJoystickY, 0.5 * gpad.leftJoystickX]
                )
        elif configuration == "4":
            if new and gpad.buttonLeft:
                if gpad.buttonA:
                    QCarCommand = np.array(
                        [-0.1 * gpad.trigger, 0.5 * gpad.leftJoystickX]
                    )
                else:
                    QCarCommand = np.array(
                        [0.1 * gpad.trigger, 0.5 * gpad.leftJoystickX]
                    )

        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])

        # Adjust LED indicators based on steering and reverse indicators based on reverse gear
        if QCarCommand[1] > 0.3:
            LEDs[0] = 1
            LEDs[2] = 1
        elif QCarCommand[1] < -0.3:
            LEDs[1] = 1
            LEDs[3] = 1
        if QCarCommand[0] < 0:
            LEDs[5] = 1

        # Perform I/O
        myCar.read_write_std(
            throttle=QCarCommand[0], steering=QCarCommand[1], LEDs=LEDs
        )

        batteryVoltage = myCar.batteryVoltage

        # Estimate linear speed in m/s
        linearSpeed = myCar.motorTach
        # encoderSpeed  = myCar.motorTach/myCar.CPS_TO_MPS
        
        # End timing this iteration
        end = elapsed_time()
        
        # --- Record Data if toggled ON ---
        if is_recording and csv_writer:
            # Added linearSpeed to the recorded row
            csv_writer.writerow([end, QCarCommand[1], linearSpeed])
        # ---------------------------------

        # Calculate computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - computationTime % sampleTime

        # Pause/sleep and print out the current timestamp
        time.sleep(sleepTime)

        if new:
            os.system("cls" if os.name == "nt" else "clear")
            recording_status = "RECORDING" if is_recording else "NOT RECORDING"
            print(
                "Car Speed:\t\t\t{0:1.2f}\tm/s\nRemaining battery capacity:\t{1:4.2f}\t%\nMotor throttle:\t\t\t{2:4.2f}\t% PWM\nSteering:\t\t\t{3:3.2f}\trad\nData Status:\t\t\t{4}".format(
                    linearSpeed,
                    100 - (batteryVoltage - 10.5) * 100 / (12.6 - 10.5),
                    QCarCommand[0],
                    QCarCommand[1],
                    recording_status
                )
            )
        timeAfterSleep = elapsed_time()
        timeStep = timeAfterSleep - start
        counter += 1

except KeyboardInterrupt:
    print("User interrupted!")

finally:
    # Ensure the file closes properly if the script is interrupted during recording
    if csv_file:
        csv_file.close()
    myCar.terminate()
    gpad.terminate()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --