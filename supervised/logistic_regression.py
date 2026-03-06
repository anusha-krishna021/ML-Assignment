import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

print("Running Logistic Regression Model...")

# Load dataset
data = pd.read_csv("train.csv")

# Features and target
X = data.drop("price_range", axis=1)
y = data["price_range"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
print("First 10 Predictions:", predictions[:10])
print("Actual Values:", y_test.values[:10])

# -------- Visualization --------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Mobile Price Category Visualization")

plt.show()
