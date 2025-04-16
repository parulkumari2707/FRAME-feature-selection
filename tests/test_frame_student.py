import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from frame.frame_selector import FRAMESelector

# Load Student Performance dataset
student_df = pd.read_csv("data/student_data_student_performance.csv")

# Define features and target
X_student: pd.DataFrame = student_df.drop(columns=["G3"], errors="ignore")  # Drop target column
y_student: pd.Series = student_df["G3"]  # Target column

# Handle missing values if any
X_student.fillna(X_student.mean(), inplace=True)
for col in X_student.select_dtypes(include=["object"]).columns:
    X_student[col].fillna(X_student[col].mode()[0], inplace=True)

# Convert categorical variables to numerical if necessary
X_student = pd.get_dummies(X_student, drop_first=True)

# Split data
X_train_student: pd.DataFrame
X_test_student: pd.DataFrame
y_train_student: pd.Series
y_test_student: pd.Series
X_train_student, X_test_student, y_train_student, y_test_student = train_test_split(
    X_student, y_student, test_size=0.2, random_state=42
)

# Apply FRAME Selector for student dataset regression
print("\n=== Running FRAME Feature Selection for Student Performance Regression ===")
regressor_model = LinearRegression()
frame_selector_student = FRAMESelector(model=regressor_model, num_features=5)
X_train_selected_student: pd.DataFrame = frame_selector_student.fit_transform(X_train_student, y_train_student)

# Print selected features and transformed data shape
print("Selected Features (Student Performance Regression):", frame_selector_student.selected_features_)
print("Transformed X_train shape:", X_train_selected_student.shape)
