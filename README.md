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

#🧪 Example Usage
```bash
from frame.frame_selector import FRAMESelector
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load data
X, y = load_iris(return_X_y=True)

# Initialize FRAME
fs = FRAMESelector(top_k=8, num_features=5, model=XGBRegressor())

# Fit and transform data
X_selected = fs.fit_transform(X, y)

# Train model
model = RandomForestClassifier()
model.fit(X_selected, y)

```
# 🛠 Parameters

| Parameter      | Type   | Description                                          |
|----------------|--------|------------------------------------------------------|
| `task`         | str    | Task type: `'classification'` or `'regression'`     |
| `top_k`        | int    | Number of top features to select                     |
| `random_state` | int    | Random seed for reproducibility                      |
| `verbose`      | bool   | If `True`, prints progress and debug information     |
| `scalers`      | bool   | Apply scaling (e.g., `StandardScaler`) before selection |
| `normalize`    | bool   | Normalize features if set to `True`                  |
| `return_scores`| bool   | Whether to return feature importance scores          |


# 📁 Project Structure
```
FRAME-FEATURE-SELECTOR/ 
├── DATA/ 
│ ├── myocardial_infarction_data.csv 
│ ├── pd_speech_features-parkinsons.csv 
│ ├── students_data_student_performance.csv 
│ └── synthetic_data/ 
│  ├── synthetic_data_with_noise.csv 
│  ├── synthetic_data_Baseline_dataset.csv 
│  ├── synthetic_data_high_sparsity_high_redundancy.csv 
│  ├── synthetic_data_high_sparsity_low_redundancy.csv 
│  ├── synthetic_data_low_sparsity_high_redundancy.csv 
│  └── synthetic_data_low_sparsity_low_redundancy.csv
├── examples/ 
├── frame/ 
│ ├── init.py 
│ └── frame_selector.py 
├── tests/ 
│ ├── test_synthetic_with_noise.py 
│ ├── test_frame_cardiovascular.py 
│ ├── test_frame_parkinsons.py 
│ ├── test_frame_regression.py 
│ ├── test_frame_classification.py 
│ ├── test_frame_script.py 
│ ├── test_frame_student.py 
│ └── model_evaluation/ 
│ └── model_evaluation.py 
├── README.md 
├── requirements.txt 
├── setup.py 
└── usage.py

```

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

# 🤝 Contributing
- Contributions are welcome! To contribute:
- Fork the repository.
- Create a new branch (git checkout -b feature-new).
- Make your changes.
- Run tests and ensure code quality.
- Submit a pull request with a clear description.

# 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

# 🌐 Connect
For suggestions, feedback, or questions, feel free to open an Issue or contact me directly.

### Happy Feature Selecting! 🎯

