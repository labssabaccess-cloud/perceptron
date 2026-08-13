from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from perceptron import Perceptron

data = load_breast_cancer()
X, y = data.data, data.target

# Split into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, random_state=42)

# Scale features (Crucial for Perceptrons to converge properly)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate your custom model
model = Perceptron(learning_rate=0.01, n_iters=1000)

# Train the model using the scaled training data
model.fit(X_train_scaled, y_train)

# Make predictions on the unseen test data
predictions = model.predict(X_test_scaled)

# Calculate accuracy percentage
accuracy = np.mean(predictions == y_test) * 100
print(f"Custom Perceptron Accuracy: {accuracy:.2f}%")
