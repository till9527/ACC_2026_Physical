import socket
import struct
import numpy as np
import cv2
import threading

# --- Settings ---
HOST = "0.0.0.0"
PORT = 8080


class VisionSteeringController:
    def __init__(self, kp=0.002):
        self.kp = kp
        self.maxSteeringAngle = np.pi / 6

    def process_image(self, image):
        h, w = image.shape[:2]
        roi = image[int(h / 2) : h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Yellow bounds (tune based on lighting)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        M = cv2.moments(mask)

        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            error = (w / 2) - cx
            return error, mask

        return 0, mask

    def update(self, image):
        error, mask = self.process_image(image)
        steering_cmd = self.kp * error
        return (
            np.clip(steering_cmd, -self.maxSteeringAngle, self.maxSteeringAngle),
            mask,
        )


def receive_all(sock, n):
    data = bytearray()
    while len(data) < n:
        try:
            sock.settimeout(5.0)
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        except (socket.timeout, socket.error):
            return None
    return data


def handle_client(conn, addr):
    print(f"[+] QCar connected from {addr}")
    visionController = VisionSteeringController(kp=0.002)
    payload_size = struct.calcsize(">L")

    try:
        while True:
            # 1. Receive Image Size Header
            packed_msg_size = receive_all(conn, payload_size)
            if not packed_msg_size:
                break
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            # 2. Receive Image Data
            frame_data = receive_all(conn, msg_size)
            if not frame_data:
                break

            # 3. Decode Image
            np_arr = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # 4. Process Vision Control
            steering_cmd, mask = visionController.update(frame)
            mask_pixels = cv2.countNonZero(mask)

            # 5. Send Command Back (Format: "delta,pixels\n")
            reply = f"{steering_cmd:.4f},{mask_pixels}\n"
            conn.sendall(reply.encode("utf-8"))

            # 6. Display Feeds
            cv2.imshow(f"QCar Feed - {addr[0]}", frame)
            cv2.imshow(f"Lane Mask - {addr[0]}", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Error handling {addr}: {e}")
    finally:
        print(f"[-] Closing connection for {addr}")
        conn.close()
        cv2.destroyAllWindows()


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Listening for QCar on {HOST}:{PORT}...")

    try:
        while True:
            conn, addr = server_socket.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.daemon = True
            thread.start()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        server_socket.close()


if __name__ == "__main__":
    main()
