import os
import glob
import cv2
import shutil
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

# Load detector + FaceNet
detector = MTCNN()
embedder = FaceNet()

dataset_path = r""
unauthorizedset_path = r""
capturedFace_path = r""
capturedFacesCount = len([f for f in os.listdir(capturedFace_path) if os.path.isfile(os.path.join(capturedFace_path, f))])
print("Captured faces count:", capturedFacesCount)

# Functions
def extract_face(img_path, required_size=(160,160)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # detect face
    results = detector.detect_faces(img)
    if len(results) == 0:
        raise ValueError("No face detected in image!")
    
    x, y, w, h = results[0]['box']  # first face
    x, y = abs(x), abs(y)
    face = img[y:y+h, x:x+w]
    
    # resize to FaceNet expected size
    face = cv2.resize(face, required_size)
    return face

def get_embedding(img_path):
    face = extract_face(img_path)
    embedding = embedder.embeddings([face])
    return embedding[0]

def align_face(image_path, save_path=None):
    """
    Detects face, aligns it based on eyes, and returns the aligned face.
    Optionally saves the aligned image if save_path is given.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return None

    # Detect faces
    results = detector.detect_faces(img)
    if len(results) == 0:
        print("No face detected!")
        return None

    # Take the first detected face
    face = results[0]
    keypoints = face['keypoints']

    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # Compute angle between the eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Find center between eyes
    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                   (left_eye[1] + right_eye[1]) // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)

    # Rotate entire image
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # Crop aligned face
    x, y, w, h = face['box']
    aligned_face = rotated[y:y+h, x:x+w]

    # Save aligned face if path given
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, aligned_face)
        print(f"Aligned face saved at {save_path}")

    return aligned_face


X, y = [], []

# Iterate through authorized faces dataset
for person in os.listdir(dataset_path):  # loops through all subfolders
    person_folder = os.path.join(dataset_path, person)
    if os.path.isdir(person_folder):  # ensure it's a folder
        for img_path in glob.glob(os.path.join(person_folder, "*.jpg")):
            X.append(get_embedding(img_path))
            y.append(person.capitalize())


X = np.array(X)
y = np.array(y)

latesttCapturedFace_path = os.path.join(capturedFace_path, "test") + str(capturedFacesCount) + ".jpg"
test = get_embedding(latesttCapturedFace_path).reshape(1, -1)


# encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)


pipeline = Pipeline([
    ('normalize', Normalizer(norm='l2')),
    ('svm', SVC(kernel="linear", probability=True))
])

pipeline.fit(X, y_enc)


# GridSearchCV Fine-tuning for large datasets
"""
params = {
    'svm__C': [1, 10, 100],   # Regularization strength
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 0.01, 0.001]
}

grid = GridSearchCV(pipeline, params, cv=3, n_jobs=-1, verbose=1)
grid.fit(X, y_enc)
print("Best Parameters:", grid.best_params_)

"""

y_pred = pipeline.predict(test)
person = le.inverse_transform(y_pred)[0]
prob = pipeline.predict_proba(test).max()

if prob < 0.6:
    shutil.copy(latesttCapturedFace_path, unauthorizedset_path)
    print(f"Predicted: Unknown Person | Confidence: {prob*100:.2f}%")
else:
    print(f"Predicted: {person} | Confidence: {prob*100:.2f}%")

