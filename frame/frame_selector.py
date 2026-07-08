import numpy as np
import pandas as pd
from typing import Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.utils.multiclass import type_of_target
from xgboost import XGBRegressor, XGBClassifier


class FRAMESelector(BaseEstimator, TransformerMixin):
    """
    FRAME Feature Selector: Combines Recursive Feature Elimination (RFE) and Forward Selection
    using an XGBoost model for selecting a subset of informative features.

    Attributes:
    ----------
    model : Optional[Union[XGBClassifier, XGBRegressor]]
        The machine learning model to use for feature selection. If None, a default XGBoost
        classifier or regressor is used based on the target variable type.

    num_features : Optional[int]
        The number of final features to select after forward selection. If None,
        defaults to half the number of features.

    top_k : int
        The number of features to retain after the initial RFE step.

    random_state : Optional[int]
        Seed for reproducibility. It seeds the default XGBoost estimator (used when
        ``model`` is None). Note: a user-supplied ``model`` is not modified -- set its
        own ``random_state`` for fully reproducible results with a custom estimator.

    selected_features_ : list[str]
        List of selected feature names after applying FRAME selection.
    """

    def __init__(
        self,
        model: Optional[Union[XGBClassifier, XGBRegressor]] = None,
        num_features: Optional[int] = None,
        top_k: int = 20,
        random_state: Optional[int] = None,
    ):
        self.model = model
        self.num_features = num_features
        self.top_k = top_k
        self.random_state = random_state

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> "FRAMESelector":
        """
        Fit the FRAME feature selector to the data.

        Parameters:
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        y : pd.Series or np.ndarray
            Target vector.

        Returns:
        -------
        self : FRAMESelector
            Fitted feature selector.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame or a NumPy array.")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series or a 1D NumPy array.")
        if hasattr(y, "ndim") and y.ndim != 1:
            raise ValueError("y must be 1-dimensional.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        if self.top_k > X.shape[1]:
            raise ValueError(
                f"top_k={self.top_k} is greater than the number of features ({X.shape[1]})."
            )

        num_features = self.num_features if self.num_features else max(1, X.shape[1] // 2)
        if num_features > self.top_k:
            raise ValueError(
                f"num_features={num_features} cannot be greater than top_k={self.top_k}."
            )

        # Record input metadata (scikit-learn convention).
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        # Determine problem type
        target_type = type_of_target(y)
        is_classification = target_type in ["binary", "multiclass"]

        # Default model selection. Never mutate the constructor argument
        # (sklearn contract): resolve the estimator into a fitted attribute.
        if self.model is None:
            self.model_ = (
                XGBClassifier(eval_metric="logloss", random_state=self.random_state)
                if is_classification
                else XGBRegressor(random_state=self.random_state)
            )
        else:
            self.model_ = clone(self.model)

        # Step 1: RFE
        rfe = RFE(estimator=self.model_, n_features_to_select=self.top_k)
        rfe.fit(X, y)
        rfe_selected_features = X.columns[rfe.support_]

        # Step 2: Forward Selection
        forward_selector = SequentialFeatureSelector(
            self.model_, n_features_to_select=num_features, direction="forward"
        )
        forward_selector.fit(X[rfe_selected_features], y)
        final_selected_features = rfe_selected_features[forward_selector.get_support()]

        self.selected_features_ = final_selected_features.tolist()
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Transform the dataset to retain only the selected features.

        Parameters:
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.

        Returns:
        -------
        X_transformed : pd.DataFrame
            Reduced feature matrix with selected features only.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame or a NumPy array.")
        if not getattr(self, "selected_features_", None):
            raise RuntimeError(
                "The FRAMESelector has not been fitted yet. Call fit() before transform()."
            )
        return X[self.selected_features_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """
        Get a mask, or integer indices, of the selected features.

        Parameters:
        ----------
        indices : bool, default=False
            If True, return the integer indices of the selected features instead
            of a boolean mask.

        Returns:
        -------
        support : np.ndarray
            Boolean mask of shape ``(n_features_in_,)``, or an integer index array
            of the selected features if ``indices=True``.
        """
        if not getattr(self, "selected_features_", None):
            raise RuntimeError(
                "The FRAMESelector has not been fitted yet. Call fit() before get_support()."
            )
        selected = set(self.selected_features_)
        mask = np.array([name in selected for name in self.feature_names_in_], dtype=bool)
        return np.flatnonzero(mask) if indices else mask

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Get output feature names for transformation.

        Parameters:
        ----------
        input_features : array-like of str or None, default=None
            Ignored; present for scikit-learn API compatibility. Output names are
            always the selected feature names determined during ``fit``.

        Returns:
        -------
        feature_names_out : np.ndarray of str
            The names of the selected features.
        """
        if not getattr(self, "selected_features_", None):
            raise RuntimeError(
                "The FRAMESelector has not been fitted yet. "
                "Call fit() before get_feature_names_out()."
            )
        return np.asarray(self.selected_features_, dtype=object)

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        """
        Fit the selector and transform the dataset in one step.

        Parameters:
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        y : pd.Series or np.ndarray
            Target vector.

        Returns:
        -------
        X_transformed : pd.DataFrame
            Transformed feature matrix with selected features.
        """
        return self.fit(X, y).transform(X)
