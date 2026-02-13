# train_gesture_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load CSV
data = pd.read_csv('gesture_data.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Accuracy
acc = model.score(X_test, y_test)
print(f"Model Accuracy: {acc*100:.2f}%")

# Save model
with open('gesture_knn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("KNN model saved as gesture_knn_model.pkl")
