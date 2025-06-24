import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Callable
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
import joblib

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPipeline:
    def __init__(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        target_column: str,
        model_name: str = 'auto',
        problem_type: str = None
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.model_name = model_name
        self.problem_type = problem_type
        self.pipeline = self._create_pipeline()
        self.best_model = None
        self.best_score = -float('inf')

        self.models = {
            'random_forest': RandomForestRegressor if problem_type == 'regression' else RandomForestClassifier,
            'linear_regression': LinearRegression if problem_type == 'regression' else LogisticRegression,
            'svm': SVR if problem_type == 'regression' else SVC,
            'xgb': XGBRegressor if problem_type == 'regression' else XGBClassifier
        }

        self.metrics = {
            'regression': {
                'r2': r2_score,
                'mse': mean_squared_error,
                'mae': mean_absolute_error
            },
            'classification': {
                'accuracy': accuracy_score,
                'precision': precision_score,
                'recall': recall_score,
                'f1': f1_score
            }
        }

        self.model_params = {
            'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
            'linear_regression': {},
            'svm': {'kernel': 'rbf'},
            'xgb': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
        }

        # Feature importance için RandomForestRegressor kullanılır
        self.feature_importance_model = RandomForestRegressor(n_estimators=100, random_state=42)

    def _create_pipeline(self) -> Pipeline:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, self.numeric_features),
            ('cat', categorical_transformer, self.categorical_features)
        ])
        return Pipeline([('preprocessor', preprocessor)])

    def _determine_problem_type(self, y: np.ndarray) -> str:
        if self.problem_type:
            return self.problem_type
        unique_values = np.unique(y)
        # Sayısal hedef ve çok sınıflı değilse regresyon, aksi halde sınıflandırma
        if pd.api.types.is_numeric_dtype(y) and len(unique_values) > 20:
            return 'regression'
        else:
            return 'classification'

    def _get_metrics(self, problem_type: str) -> Dict[str, Callable]:
        return self.metrics[problem_type]

    def _get_model(self, model_name: str, problem_type: str) -> Any:
        return self.models[model_name]

    def _get_model_params(self, model_name: str) -> Dict[str, Any]:
        return self.model_params[model_name]

    def get_feature_names(self) -> List[str]:
        preprocessor = self.pipeline.named_steps['preprocessor']
        feature_names = []

        # Sayısal özellikler
        if 'num' in preprocessor.named_transformers_:
            feature_names.extend(self.numeric_features)

        # Kategorik özellikler (OneHotEncoder sonrası)
        if 'cat' in preprocessor.named_transformers_:
            cat_transformer = preprocessor.named_transformers_['cat']
            ohe = cat_transformer.named_steps['onehot']
            ohe_features = ohe.get_feature_names_out(self.categorical_features)
            feature_names.extend(ohe_features.tolist())

        return feature_names

    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        self.feature_importance_model.fit(X, y)
        importance = self.feature_importance_model.feature_importances_
        feature_names = self.get_feature_names()
        return dict(zip(feature_names, importance))

    def auto_train(self, df: pd.DataFrame) -> Dict[str, Any]:
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        problem_type = self._determine_problem_type(y)
        logger.info(f"Problem tipi belirlendi: {problem_type}")

        X_processed = self.pipeline.fit_transform(X)
        metrics = self._get_metrics(problem_type)
        model_list = ['random_forest', 'linear_regression', 'svm', 'xgb']

        best_model = None
        best_score = -float('inf')
        best_model_name = None
        best_metrics = {}

        for model_name in model_list:
            model_class = self._get_model(model_name, problem_type)
            model_params = self._get_model_params(model_name)
            model = model_class(**model_params)
            model.fit(X_processed, y)
            y_pred = model.predict(X_processed)

            if problem_type == 'regression':
                metric_results = {
                    'r2': float(metrics['r2'](y, y_pred)),
                    'mse': float(metrics['mse'](y, y_pred)),
                    'mae': float(metrics['mae'](y, y_pred))
                }
                score = metric_results['r2']
            else:
                metric_results = {
                    'accuracy': float(metrics['accuracy'](y, y_pred)),
                    'precision': float(metrics['precision'](y, y_pred, average='weighted')),
                    'recall': float(metrics['recall'](y, y_pred, average='weighted')),
                    'f1': float(metrics['f1'](y, y_pred, average='weighted'))
                }
                score = metric_results['accuracy']

            logger.info(f"{model_name} modeli için skor: {score}")

            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = model_name
                best_metrics = metric_results

        self.best_model = best_model
        self.best_score = best_score

        feature_importance = self._calculate_feature_importance(X_processed, y)

        return {
            'model_type': problem_type,
            'best_model': best_model_name,
            'metrics': best_metrics,
            'feature_importance': feature_importance
        }

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("No trained model available")
        processed = self.pipeline.transform(data)
        return self.best_model.predict(processed)
