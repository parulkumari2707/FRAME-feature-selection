import numpy as np
import pandas as pd
import time
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_squared_error
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_classif
from sklearn.preprocessing import label_binarize
from frame.frame_selector import FRAMESelector

random_state = 42

# ==================== FEATURE SELECTION FUNCTIONS ====================
def rfe_selection(X: pd.DataFrame, y: pd.Series, num_features: int = 5) -> np.ndarray:
    """Selects features using Recursive Feature Elimination (RFE)."""
    if num_features > X.shape[1]:
        raise ValueError("num_features cannot be greater than the number of features in X")
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    selector = RFE(model, n_features_to_select=num_features)
    selector.fit(X, y)
    return selector.support_

def k_best_selection(X: pd.DataFrame, y: pd.Series, num_features: int = 5) -> np.ndarray:
    """Selects top k features based on univariate statistical tests."""
    if num_features > X.shape[1]:
        raise ValueError("num_features cannot be greater than the number of features in X")
    selector = SelectKBest(score_func=f_regression, k=num_features)
    selector.fit(X, y)
    return selector.get_support()

def lasso_selection(X: pd.DataFrame, y: pd.Series, num_features: int = 5, alpha: float = 0.02) -> np.ndarray:
    """Selects features using Lasso regularization."""
    if num_features > X.shape[1]:
        raise ValueError("num_features cannot be greater than the number of features in X")
    model = Lasso(alpha=alpha, random_state=random_state)
    model.fit(X, y)
    coef = np.abs(model.coef_)
    top_indices = np.argsort(coef)[-num_features:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_indices] = True
    return mask

def mutual_information_selection(X: pd.DataFrame, y: pd.Series, num_features: int = 5) -> np.ndarray:
    """Selects features based on mutual information scores."""
    if num_features > X.shape[1]:
        raise ValueError("num_features cannot be greater than the number of features in X")
    if y.ndim > 1:
        y = y.ravel()
    mi = mutual_info_classif(X, y)
    top_indices = np.argsort(mi)[-num_features:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_indices] = True
    return mask

def tree_based_selection(X: pd.DataFrame, y: pd.Series, num_features: int = 5) -> np.ndarray:
    """Selects features based on tree-based feature importances."""
    if num_features > X.shape[1]:
        raise ValueError("num_features cannot be greater than the number of features in X")
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X, y)
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-num_features:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_indices] = True
    return mask

def frame_selection(X: pd.DataFrame, y: pd.Series, num_features: int = 5) -> np.ndarray:
    """Selects features using the FRAME feature selection method."""
    if num_features > X.shape[1]:
        raise ValueError("num_features cannot be greater than the number of features in X")
    try:
        selector = FRAMESelector(model=LogisticRegression(max_iter=1000), num_features=num_features)
        selector.fit(X, y)
        selected = selector.selected_features_
        indices = [X.columns.get_loc(col) for col in selected]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[indices] = True
        return mask
    except Exception as e:
        print("FRAME Error:", e)
        return np.zeros(X.shape[1], dtype=bool)

methods: Dict[str, callable] = {
    "FRAME": frame_selection,
    "RFE": rfe_selection,
    "SelectKBest": k_best_selection,
    "Lasso": lasso_selection,
    "Mutual Information": mutual_information_selection,
    "Tree-based Feature Importance": tree_based_selection,
}

# ==================== MODELS ====================
models: Dict[str, object] = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
    "Random Forest": RandomForestClassifier(n_estimators=500, random_state=random_state),
    "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    "AdaBoost": AdaBoostClassifier(random_state=random_state),
    "SVM": SVC(kernel='linear', probability=True, random_state=random_state),
}

# ==================== MAIN RUNNER ====================
def run_test_dataset(name: str, filepath: str, target_col: Optional[str], header: int = 0, num_features: int = 5) -> None:
    """
    Runs evaluation on the specified dataset using multiple feature selection methods and models.

    Parameters:
    - name: Name of the dataset
    - filepath: File path to the dataset CSV
    - target_col: Target column name
    - header: CSV header row (default: 0)
    - num_features: Number of features to select for evaluation
    """
    print(f"\n--- Running Dataset: {name} ---")
    try:
        df = pd.read_csv(filepath, header=header)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        if target_col is not None and target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset '{name}'.")

        X = df.drop(columns=[target_col], errors='ignore')
        y = df[target_col] if target_col else None

        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]
            elif y.shape[1] > 1:
                y = y.idxmax(axis=1)

        y = np.ravel(y)

        for col in X.select_dtypes(include=['object']).columns:
            X[col].fillna(X[col].mode()[0], inplace=True)
        X.fillna(X.mean(), inplace=True)

        X = pd.get_dummies(X, drop_first=True)

        is_classification = False
        if pd.api.types.is_object_dtype(y) or (np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) < 20):
            is_classification = True
            y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        results = []
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        for method_name, method in methods.items():
            print(f"🔍 Feature Selection Method: {method_name}")
            try:
                start_time = time.time()
                mask = method(X_train_scaled, y_train, num_features=num_features)
                selected_features = X_train_scaled.columns[mask]

                if len(selected_features) == 0:
                    print(f"⚠️  No features selected by {method_name}, skipping...")
                    continue

                X_train_selected = X_train_scaled[selected_features]
                X_test_selected = X_test_scaled[selected_features]

                for model_name, model in models.items():
                    model.fit(X_train_selected, y_train)
                    y_pred = model.predict(X_test_selected)

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = recall = f1 = auc = r2 = mse = nrmse = np.nan

                    if is_classification:
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        if hasattr(model, "predict_proba"):
                            y_proba = model.predict_proba(X_test_selected)
                            if n_classes > 2:
                                y_bin = label_binarize(y_test, classes=unique_classes)
                                auc = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
                            else:
                                auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        nrmse = np.sqrt(mse) / (y_test.max() - y_test.min())

                    results.append({
                        "Dataset": name,
                        "Feature Selection Method": method_name,
                        "Model": model_name,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1,
                        "AUC-ROC": auc,
                        "R2 Score": r2,
                        "MSE": mse,
                        "NRMSE": nrmse,
                        "Time Taken (seconds)": time.time() - start_time,
                        "Number of Features Selected": len(selected_features),
                        "Selected Features": ", ".join(selected_features[:10]) + ("..." if len(selected_features) > 10 else ""),
                    })
            except Exception as fe:
                print(f"❌ Error in {method_name}: {fe}")
                continue

        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(f"results_{name.replace(' ', '_').lower()}.csv", index=False)
            print(f"✅ Results stored for {name}")
        else:
            print(f"⚠️ No results to store for {name}")

    except Exception as e:
        print(f"❌ Failed to process {name}: {e}")

# ==================== DATASETS TO EVALUATE ====================
datasets: Dict[str, Tuple[str, Optional[str], int]] = {
    "Parkinsons 1": ("data/pd_speech_features_parkinsons.csv", "class", 1),
    "Student Performance 1": ("data/student_data_student_performance.csv", "G3", 0),
    "Cardiovascular 1": ("data/myocardial_infarction_data.csv", "LET_IS", 0),
    "Synthetic Baseline 1 ": ("data/synthetic_data/synthetic_dataset_Baseline_dataset.csv", "target", 0),
    "Synthetic High Sparse High Redundancy 1": ("data/synthetic_data/synthetic_dataset_High_sparsity,_high_redundancy.csv", "target", 0),
    "Synthetic High Sparse Low Redundancy 1": ("data/synthetic_data/synthetic_dataset_High_sparsity,_low_redundancy.csv", "target", 0),
    "Synthetic Low Sparse High Redundancy 1": ("data/synthetic_data/synthetic_dataset_Low_sparsity,_high_redundancy.csv", "target", 0),
    "Synthetic Low Sparse Low Redundancy 1": ("data/synthetic_data/synthetic_dataset_Low_sparsity,_low_redundancy.csv", "target", 0),
}

for name, (filepath, target, header) in datasets.items():
    run_test_dataset(name, filepath, target, header)
