import cv2

def list_available_cameras(max_cameras=5):
    """Check available camera indexes."""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def main():
    print("🔍 Detecting available cameras...")
    cameras = list_available_cameras()

    if not cameras:
        print("❌ No camera detected.")
        return

    print(f"✅ Available cameras: {cameras}")
    cam_index = int(input(f"Select camera index from {cameras}: "))

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    print("🎥 Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        cv2.imshow(f"Camera {cam_index}", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
