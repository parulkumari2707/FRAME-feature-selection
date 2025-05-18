# FRAME-FEATURE-SELECTOR

FRAME-FEATURE-SELECTOR is a Python library that implements **FRAME** (Forward Recursive Adaptive Model Extraction), a robust and interpretable feature selection technique for both classification and regression tasks. It allows practitioners and researchers to compare FRAME with other traditional feature selection methods and evaluate its performance on various datasets.

---

## 🧠 What is FRAME?

**FRAME** is a hybrid feature selection method proposed in the [FRAME paper on arXiv](https://arxiv.org/abs/2501.11972) that aggregates feature importance scores across multiple traditional techniques and model evaluations. Instead of relying on a single feature selector, FRAME combines the strengths of Forward Feature Selection and RFE(Recursive Feature selection) with XGBoost as estimator to produce a ranked list of features. This approach reduces bias, improves generalizability, and offers more reliable performance across diverse datasets. It aggregates feature importance scores across multiple traditional techniques using recursive evaluation loops.

---

## 📦 Installation

- To install FRAME-FEATURE-SELECTOR from source:

```bash
git clone https://github.com/parulkumari2707@gmail.com/FRAME-FEATURE-SELECTOR.git
cd FRAME-FEATURE-SELECTOR
pip install -e
```

- To install from PyPI:
 ```bash
 pip install frame-feature-selector
```

# 🚀 Key Features
- 🔍 Hybrid feature selection using multiple evaluators.
- 🧪 Works for both classification and regression tasks.
- 📊 Evaluates and benchmarks multiple feature selectors including FRAME.
- 📁 Supports real-world and synthetic datasets.
- 📈 Outputs detailed performance metrics (Accuracy, F1, ROC-AUC, R², MSE, etc.).
- 📂 Modular and extensible design with scikit-learn-style API.
- 🧪 Built-in testing framework and dataset pipeline.

# ⚙️ How It Works
FRAME:
- Applies multiple feature selection techniques on a given dataset.
- Ranks features from each technique and aggregates them into a unified ranking.
- Selects the top-k (or thresholded) features for downstream model training.
- Evaluates and compares model performance across selectors.

# 🧪 Example Usage

### Classification Example (Cardiovascular Data)
```bash
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from frame.frame_selector import FRAMESelector

# Load data
df = pd.read_csv("data/myocardial_infarction_data.csv")
X = df.drop(columns=["LET_IS"], errors='ignore')
y = df["LET_IS"]

# Handle missing values
X.fillna(X.mean(), inplace=True)
for col in X.select_dtypes(include=["object"]).columns:
    X[col].fillna(X[col].mode()[0], inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize FRAME with XGBClassifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
frame_selector = FRAMESelector(model=model, num_features=5)

# Fit and transform data
X_selected = frame_selector.fit_transform(X_train, y_train)

# Check selected features
print("Selected Features:", frame_selector.selected_features_)
```
### Regression Example (Student Performance Data)
```bash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from frame.frame_selector import FRAMESelector

# Load Student Performance dataset
student_df = pd.read_csv("data/student_data_student_performance.csv")

# Define features and target
X = student_df.drop(columns=["G3"], errors="ignore")  # Drop target column
y = student_df["G3"]  # Target column

# Handle missing values if any
X.fillna(X.mean(), inplace=True)
for col in X.select_dtypes(include=["object"]).columns:
    X[col].fillna(X[col].mode()[0], inplace=True)

# Convert categorical variables to numerical
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply FRAME Selector with Linear Regression
regressor_model = LinearRegression()
frame_selector = FRAMESelector(model=regressor_model, num_features=5)
X_selected = frame_selector.fit_transform(X_train, y_train)

# Print selected features
print("Selected Features:", frame_selector.selected_features_)
print("Transformed X shape:", X_selected.shape)
```

# 🛠 Parameters

| Parameter      | Type      | Description                                                                 |
|----------------|-----------|-----------------------------------------------------------------------------|
| model          | object    | Base estimator (e.g., `XGBClassifier`, `LinearRegression`, etc.)            |
| num_features   | int       | Final number of features to select                                          |
| top_k          | int       | Number of top features to keep after initial filtering                      |
| task           | str       | Task type: `'classification'` or `'regression'` (auto-detected if not specified) |
| random_state   | int       | Random seed for reproducibility                                             |
| verbose        | bool      | If `True`, prints progress and debug information                            |
| scalers        | bool      | Apply scaling (e.g., `StandardScaler`) before selection                     |
| normalize      | bool      | Normalize features if set to `True`                                         |
| return_scores  | bool      | Whether to return feature importance sc                                     |

# 📋 Requirements
- Python ≥ 3.7
- NumPy
- pandas
- scikit-learn
- scipy

# Install dependencies via:
``` bash
pip install -r requirements.txt
```

# 🧪 Running Tests
To run the test suite:
```bash pytest tests/ ```

### To run specific tests:
```bash
# For cardiovascular dataset tests
pytest tests/test_frame_cardiovascular.py

# For student performance regression tests
pytest tests/test_frame_student.py

# For general regression functionality tests
pytest tests/test_frame_regression.py
```

# 🤝 Contributing
Contributions are welcome! To contribute:
- Fork the repository.
- Create a new branch (git checkout -b feature-new).
- Make your changes.
- Run tests and ensure code quality.
- Submit a pull request with a clear description.

# 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

# 🌐 Connect
For suggestions, feedback, or questions, feel free to open an Issue or contact me directly.

# Happy Feature Selecting! 🎯

