import subprocess
import cv2
import os
import time
from datetime import datetime

SAVE_DIR = "/home/pi/noir_photos"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Press ENTER to start capturing faces for 5 seconds...")

while True:
    key = input("Press ENTER to capture, or type 'q' to quit: ")
    if key.lower() == 'q':
        break
    
    start_time = time.time()
    while time.time() - start_time < 5:  # capture for 5 seconds
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(SAVE_DIR, f"frame_{timestamp}.jpg")
        
        # Capture image using rpicam-hello (short preview, then save)
        subprocess.run([
            "rpicam-hello",
            "--output", filename,
            "--timeout", "100"  # 0.1 second preview
        ])
        
        # Read image with OpenCV
        frame = cv2.imread(filename)
        if frame is None:
            continue  # skip if image not captured
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Draw bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
print("Done capturing faces.")