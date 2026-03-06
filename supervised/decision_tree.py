import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Running Decision Tree Model...")

# Load dataset
data = pd.read_csv("train.csv")

# Features and target
X = data.drop("price_range", axis=1)
y = data["price_range"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
print("First 10 Predictions:", predictions[:10])
print("Actual Values:", y_test.values[:10])

# -------- Visualization --------
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X.columns,
          class_names=["Low", "Medium", "High", "Very High"])
plt.title("Decision Tree for Mobile Price Classification")
plt.show()
