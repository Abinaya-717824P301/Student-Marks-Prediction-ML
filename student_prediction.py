import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student_marks.csv")

# Display dataset
print("Dataset:")
print(data)

# Visualization
plt.scatter(data['StudyHours'], data['Marks'])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.savefig("graph.png")  # Saves graph as image
plt.close()

# Features and target
X = data[['StudyHours', 'Attendance', 'Assignments']]
y = data['Marks']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate accuracy
accuracy = model.score(X_test, y_test)

# New student data for prediction
new_student = pd.DataFrame({
    'StudyHours': [6],
    'Attendance': [90],
    'Assignments': [7]
})

# Predict marks
prediction = model.predict(new_student)

# Output results
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("Predicted Marks:", round(prediction[0], 2))
