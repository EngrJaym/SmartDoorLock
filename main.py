import os
import glob
import cv2
import time
import csv
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from datetime import datetime
from dotenv import load_dotenv
from picamera2 import Picamera2

load_dotenv()
# --- PATHS ---
dataset_path = os.getenv('dataset_path')
captured_faces_path = os.getenv('captured_faces_path')
unauthorized_faces_path = os.getenv('unauthorized_faces_path')

log_all = "logs_all.csv"
log_auth = "logs_authorized.csv"
log_unauth = "logs_unauthorized.csv"

# Ensure directories exist
os.makedirs(captured_faces_path, exist_ok=True)
os.makedirs(unauthorized_faces_path, exist_ok=True)

# --- FUNCTIONS ---
def extract_face_from_frame(frame, required_size=(160,160)):
    # MTCNN expects RGB, so convert if needed
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        rgb = frame # frame is already RGB if captured correctly with Picamera2
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    results = detector.detect_faces(rgb)
    if len(results) == 0:
        return None
    x, y, w, h = results[0]['box']
    x, y = abs(x), abs(y)
    face = rgb[y:y+h, x:x+w]
    face = cv2.resize(face, required_size)
    return face

def get_embedding_from_face(face):
    embedding = embedder.embeddings([face])
    return embedding[0]

def init_csv(file):
    if not os.path.exists(file):
        with open(file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "person", "probability", "image_path"])

def log_event(file, person, prob, image_path=""):
    with open(file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), person, f"{prob*100:.2f}%", image_path])

# --- INIT DETECTOR + FACENET ---
detector = MTCNN()
embedder = FaceNet()

# --- TRAINING ON AUTHORIZED DATASET ---
print("📊 Training model on authorized faces...")
X, y = [], []
for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)
    if os.path.isdir(person_folder):
        for img_path in glob.glob(os.path.join(person_folder, "*.jpg")):
            img = cv2.imread(img_path)
            # Ensure images loaded from disk are in RGB for consistency
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face = extract_face_from_frame(img)
                if face is not None:
                    X.append(get_embedding_from_face(face))
                    y.append(person.capitalize())

X, y = np.array(X), np.array(y)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train pipeline
pipeline = Pipeline([
    ('normalize', Normalizer(norm='l2')),
    ('svm', SVC(kernel="linear", probability=True))
])
pipeline.fit(X, y_enc)

print("✅ Training complete! Starting live camera...")

# --- CSV INIT ---
for f in [log_all, log_auth, log_unauth]:
    init_csv(f)

# --- STATE TRACKERS ---
last_person = None
last_seen_time = {}   # dict: {person: timestamp}

COOLDOWN = 10  # seconds

# --- LIVE CAMERA - PICAMERA2 SETUP ---
picam2 = Picamera2()
# For best compatibility with OpenCV, specify XRGB8888 format
camera_config = picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

# Wait for autofocus to settle if using a Camera Module 3 with autofocus
time.sleep(2)
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

try:
    while True:
        # Capture a frame directly into a NumPy array
        frame = picam2.capture_array()
        
        # NOTE: MTCNN expects RGB, but OpenCV functions like cv2.rectangle expect BGR.
        # So we use the original RGB frame for face detection, and convert to BGR for drawing.
        rgb_frame = frame
        
        results = detector.detect_faces(rgb_frame)

        for result in results:
            x, y, w, h = result['box']
            x, y = abs(x), abs(y)
            face = rgb_frame[y:y+h, x:x+w]

            try:
                face_resized = cv2.resize(face, (160, 160))
                embedding = get_embedding_from_face(face_resized).reshape(1, -1)

                y_pred = pipeline.predict(embedding)
                person = le.inverse_transform(y_pred)[0]
                prob = pipeline.predict_proba(embedding).max()

                # Low confidence = Unauthorized
                if prob < 0.6:
                    person = "Unauthorized"
                    color = (0, 0, 255)
                    label = f"Unknown ({prob*100:.1f}%)"
                else:
                    color = (0, 255, 0)
                    label = f"{person} ({prob*100:.1f}%)"

                now = time.time()
                should_trigger = False

                # Trigger if different person OR cooldown expired
                if person != last_person:
                    should_trigger = True
                elif person in last_seen_time and (now - last_seen_time[person] >= COOLDOWN):
                    should_trigger = True

                if should_trigger:
                    last_person = person
                    last_seen_time[person] = now  # update timestamp

                    if person == "Unauthorized":
                        # Save full frame (convert to BGR for saving with cv2)
                        filename = f"unauth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        save_path = os.path.join(unauthorized_faces_path, filename)
                        cv2.imwrite(save_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

                        # Save cropped face (already in RGB)
                        face_filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        face_path = os.path.join(captured_faces_path, face_filename)
                        cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

                        # Logs
                        log_event(log_unauth, person, prob, save_path)
                        log_event(log_all, person, prob, save_path)

                    else:
                        # Save cropped face (already in RGB)
                        face_filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        face_path = os.path.join(captured_faces_path, face_filename)
                        cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

                        # Logs
                        log_event(log_auth, person, prob, face_path)
                        log_event(log_all, person, prob, face_path)

                # Draw bounding box on the BGR frame for display
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                cv2.rectangle(bgr_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(bgr_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            except Exception as e:
                print("Error processing face:", e)

        # Convert the frame to BGR for display with OpenCV
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Recognition", bgr_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()

