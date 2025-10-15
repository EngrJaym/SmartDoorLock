from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd


dataset = load_iris()
X = dataset.data
y = dataset.target

model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(kernel="linear", probability=True))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
results = pd.DataFrame({
    "Actual": pd.Series(y_test),
    "Predicted": y_pred,
    "Confidence": [value*100 for value in y_prob.max(axis=1)]
})

print("\n=== Predictions ===")
print(results)

# --- Metrics ---
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n=== Metrics ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-score (weighted): {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n=== Confusion Matrix ===")
print(cm)

# Classification Report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=dataset.target_names))

"""
housing = fetch_california_housing()
X = housing.data
y = housing.target

model = Pipeline([
    ("scaler", StandardScaler()),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

new_house = [[8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23]]

prediction = model.predict(new_house)

print(f"Predicted Price: ${prediction[0]*100000:,.2f}")"""