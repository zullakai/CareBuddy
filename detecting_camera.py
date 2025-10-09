import cv2

def list_available_cameras(max_cameras=5):
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def open_all_cameras(camera_indexes):
    """Open all cameras and store them in a dictionary."""
    caps = {}
    for i in camera_indexes:
        cap = cv2.VideoCapture(i)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        caps[i] = cap
        print(f"✅ Camera {i} opened.")
    return caps

def main():
    print("🔍 Detecting available cameras...")
    cameras = list_available_cameras()
    if not cameras:
        print("❌ No camera detected.")
        return

    print(f"✅ Available cameras: {cameras}")

    caps = open_all_cameras(cameras)
    current_index = cameras[0]
    print(f"🎥 Showing camera {current_index}. Press number (0–9) to switch, 'q' to quit.")

    while True:
        cap = caps[current_index]
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Failed to read frame from camera {current_index}")
            continue

        cv2.putText(frame, f"Camera {current_index}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera Viewer", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if 48 <= key <= 57:  # Number keys
            selected = key - 48
            if selected in cameras:
                current_index = selected
                print(f"🔄 Switched to camera {current_index}")
            else:
                print(f"⚠️ Camera {selected} not available")

    # Cleanup
    for c in caps.values():
        c.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
