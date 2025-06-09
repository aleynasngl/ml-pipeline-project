import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Callable
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
import joblib
import mlflow
from mlflow.models import infer_signature
import json

# Configure logging
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
        """
        Pipeline'ı başlat
        
        Args:
            numeric_features (List[str]): Sayısal özellikler
            categorical_features (List[str]): Kategorik özellikler
            target_column (str): Hedef sütun
            model_name (str): Model adı ('auto', 'random_forest', 'linear_regression', 'svm', 'xgb')
            problem_type (str): Problem tipi ('classification' veya 'regression')
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.model_name = model_name
        self.problem_type = problem_type
        
        # Pipeline'ı oluştur
        self.pipeline = self._create_pipeline()
        
        # En iyi modeli sakla
        self.best_model = None
        self.best_score = -float('inf')
        
        # Model sözlüğü
        self.models = {
            'random_forest': RandomForestRegressor if problem_type == 'regression' else RandomForestClassifier,
            'linear_regression': LinearRegression if problem_type == 'regression' else LogisticRegression,
            'svm': SVR if problem_type == 'regression' else SVC,
            'xgb': XGBRegressor if problem_type == 'regression' else XGBClassifier
        }
        
        # Metrik sözlüğü
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
        
        # Model parametreleri
        self.model_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'linear_regression': {
                # random_state parametresi kaldırıldı
            },
            'svm': {
                'kernel': 'rbf'
                # random_state parametresi kaldırıldı
            },
            'xgb': {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42
            }
        }
        
        # Feature importance için model
        self.feature_importance_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # MLflow ayarları - devre dışı bırakıldı
        # mlflow.set_tracking_uri("http://localhost:5000")
        # mlflow.set_experiment("ml_pipeline")
        
    def _create_pipeline(self) -> Pipeline:
        """Veri ön işleme pipeline'ını oluştur"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return Pipeline(steps=[('preprocessor', preprocessor)])
        
    def _determine_problem_type(self, y: np.ndarray) -> str:
        """
        Hedef değişkenin tipine göre problem tipini belirle
        
        Args:
            y (np.ndarray): Hedef değişken
            
        Returns:
            str: Problem tipi ('classification' veya 'regression')
        """
        # Eğer problem tipi zaten belirlenmişse onu kullan
        if self.problem_type:
            return self.problem_type
            
        # Hedef değişkenin benzersiz değer sayısını kontrol et
        unique_values = np.unique(y)
        
        # Eğer hedef değişken kategorik ise veya benzersiz değer sayısı 10'dan az ise classification
        if len(unique_values) <= 10:
            return 'classification'
        # Eğer hedef değişken sayısal ise ve benzersiz değer sayısı 10'dan fazla ise regression
        elif len(unique_values) > 10 and np.issubdtype(y.dtype, np.number):
            return 'regression'
        # Varsayılan olarak classification
        else:
            return 'classification'
            
    def _get_metrics(self, problem_type: str) -> Dict[str, Callable]:
        """
        Problem tipine göre metrikleri döndür
        
        Args:
            problem_type (str): Problem tipi ('classification' veya 'regression')
            
        Returns:
            Dict[str, Callable]: Metrik sözlüğü
        """
        return self.metrics[problem_type]
        
    def _get_model(self, model_name: str, problem_type: str) -> Any:
        """
        Model adına göre model sınıfını döndür
        
        Args:
            model_name (str): Model adı
            problem_type (str): Problem tipi
            
        Returns:
            Any: Model sınıfı
        """
        return self.models[model_name]
        
    def _get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Model adına göre parametreleri döndür
        
        Args:
            model_name (str): Model adı
            
        Returns:
            Dict[str, Any]: Model parametreleri
        """
        return self.model_params[model_name]
        
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Feature importance hesapla
        
        Args:
            X (np.ndarray): Özellik matrisi
            y (np.ndarray): Hedef değişken
            
        Returns:
            Dict[str, float]: Feature importance sözlüğü
        """
        # Feature importance modelini eğit
        self.feature_importance_model.fit(X, y)
        
        # Feature importance değerlerini al
        importance = self.feature_importance_model.feature_importances_
        
        # Özellik isimlerini al
        feature_names = self.numeric_features + self.categorical_features
        
        # Feature importance sözlüğünü oluştur
        feature_importance = dict(zip(feature_names, importance))
        
        return feature_importance
        
    def auto_train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Otomatik model eğitimi yap
        
        Args:
            df (pd.DataFrame): Veri seti
            
        Returns:
            Dict[str, Any]: Eğitim sonuçları
        """
        # Veriyi ayır
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Problem tipini belirle
        problem_type = self._determine_problem_type(y)
        logger.info(f"Problem tipi belirlendi: {problem_type}")
        
        # Veriyi ön işle
        X_processed = self.pipeline.fit_transform(X)
        logger.info("Veri ön işleme tamamlandı")
        
        # Metrikleri al
        metrics = self._get_metrics(problem_type)
        
        # Model listesi
        model_list = ['random_forest', 'linear_regression', 'svm', 'xgb']
        
        # En iyi modeli bul
        best_model = None
        best_score = -float('inf')
        best_model_name = None
        
        # Her modeli dene
        for model_name in model_list:
            logger.info(f"Model deneniyor: {model_name}")
            
            # Model sınıfını al
            model_class = self._get_model(model_name, problem_type)
            
            # Model parametrelerini al
            model_params = self._get_model_params(model_name)
            
            # Modeli oluştur
            model = model_class(**model_params)
            
            # Modeli eğit
            model.fit(X_processed, y)
            
            # Tahmin yap
            y_pred = model.predict(X_processed)
            
            # Metrikleri hesapla
            if problem_type == 'regression':
                score = metrics['r2'](y, y_pred)
            else:
                score = metrics['accuracy'](y, y_pred)
            
            # En iyi modeli güncelle
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = model_name
                logger.info(f"Yeni en iyi model: {model_name} (Skor: {score:.4f})")
        
        # En iyi modeli kaydet
        self.best_model = best_model
        self.best_score = best_score
        
        # Feature importance hesapla
        feature_importance = self._calculate_feature_importance(X_processed, y)
        
        # Sonuçları döndür
        return {
            'model_type': problem_type,
            'best_model': best_model_name,
            'metrics': {
                'score': best_score
            },
            'feature_importance': feature_importance
        }

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Veriyi ön işler ve hazırlar
        
        Args:
            data (pd.DataFrame): Ham veri
            
        Returns:
            pd.DataFrame: Ön işlenmiş veri
        """
        try:
            # Hedef sütundaki NaN değerleri temizle
            if data[self.target_column].isnull().any():
                data = data.dropna(subset=[self.target_column])
                logger.info(f"Hedef sütundaki NaN değerler temizlendi. Kalan satır sayısı: {len(data)}")
            
            # Eksik değerleri doldur
            for col in self.numeric_features:
                if data[col].isnull().any():
                    data[col] = data[col].fillna(data[col].median())
            
            for col in self.categorical_features:
                if data[col].isnull().any():
                    data[col] = data[col].fillna(data[col].mode()[0])
            
            # Aykırı değerleri işle (IQR yöntemi ile)
            for col in self.numeric_features:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[col] = data[col].clip(lower_bound, upper_bound)
            
            logger.info("Veri ön işleme tamamlandı")
            return data
            
        except Exception as e:
            logger.error(f"Veri ön işleme sırasında hata: {str(e)}")
            raise

    def get_scaler(self):
        """Get the appropriate scaler based on scaler_type"""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        return scalers.get(self.scaler_type, StandardScaler())
    
    def get_imputer(self):
        """Get the appropriate imputer based on imputer_type"""
        imputers = {
            'simple': SimpleImputer(strategy='median'),
            'knn': KNNImputer(n_neighbors=5)
        }
        return imputers.get(self.imputer_type, SimpleImputer(strategy='median'))

    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline for numeric and categorical features
        
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        numeric_transformer = Pipeline(steps=[
            ('imputer', self.get_imputer()),
            ('scaler', self.get_scaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return preprocessor
    
    def get_model_and_params(self) -> Tuple[Any, Dict[str, List[Any]]]:
        """
        Get model class and parameter grid for hyperparameter tuning
        
        Returns:
            Tuple[Any, Dict[str, List[Any]]]: Model class and parameter grid
        """
        if self.model_name not in self.models:
            raise ValueError(f"Invalid model name: {self.model_name}")
        
        return self.models[self.model_name], self.model_params[self.model_name]
    
    def create_pipeline(self) -> Tuple[Pipeline, Dict[str, List[Any]]]:
        """
        Create the complete ML pipeline
        
        Returns:
            Tuple[Pipeline, Dict[str, List[Any]]]: Complete ML pipeline and parameter grid
        """
        preprocessor = self.create_preprocessing_pipeline()
        model_class, param_grid = self.get_model_and_params()
        
        steps = [('preprocessor', preprocessor)]
        
        steps.append(('model', model_class(**param_grid)))
        
        pipeline = Pipeline(steps)
        
        return pipeline, param_grid

    def train(
        self,
        data: pd.DataFrame,
        cv: int = 5,
        scoring: str = None,
        use_mlflow: bool = False  # MLflow'u varsayılan olarak devre dışı bırak
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            data (pd.DataFrame): Training data
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            use_mlflow (bool): Whether to use MLflow for tracking
            
        Returns:
            Dict[str, Any]: Training results
        """
        try:
            # Preprocess data
            data = self.preprocess_data(data)
            
            # Determine problem type if not specified
            if self.problem_type is None:
                self.problem_type = self._determine_problem_type(data[self.target_column].values)
            
            # If model_name is 'auto', use auto_train
            if self.model_name == 'auto':
                logger.info("Otomatik model seçimi başlatılıyor...")
                return self.auto_train(data)
            
            # Split features and target
            X = data.drop(columns=[self.target_column])
            y = data[self.target_column]
            
            # Create pipeline
            pipeline, param_grid = self.create_pipeline()
            
            # Set default scoring if not specified
            if scoring is None:
                scoring = 'accuracy' if self.problem_type == 'classification' else 'r2'
            
            # Perform grid search
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            # Save pipeline and metrics
            self.pipeline = grid_search.best_estimator_
            self.metrics = self._calculate_metrics(X, y)
            self.feature_importance = self._calculate_feature_importance(X.values, y.values)
            
            return {
                'problem_type': self.problem_type,
                'metrics': self.metrics,
                'best_params': grid_search.best_params_,
                'feature_importance': self.feature_importance
            }
                
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise

    def _calculate_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calculate model performance metrics
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        if self.pipeline is None:
            raise ValueError("Model henüz eğitilmemiş")
            
        y_pred = self.pipeline.predict(X)
        
        if self.problem_type == 'classification':
            metrics = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'precision': float(precision_score(y, y_pred, average='weighted')),
                'recall': float(recall_score(y, y_pred, average='weighted')),
                'f1': float(f1_score(y, y_pred, average='weighted'))
            }
        else:  # regression
            metrics = {
                'mse': float(mean_squared_error(y, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
                'r2': float(r2_score(y, y_pred))
            }
        
        return metrics
    
    def save_model(self, version: str = None) -> str:
        """
        Save the trained model and metrics
        
        Args:
            version (str, optional): Model version. Defaults to timestamp.
            
        Returns:
            str: Path to saved model
        """
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create save directory
        save_dir = os.path.join('models', self.problem_type, self.model_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, f'{version}.joblib')
        joblib.dump(self.best_model, model_path)
        
        # Save metrics
        metrics_path = os.path.join(save_dir, f'{version}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        logger.info(f"Model and metrics saved to {save_dir}")
        
        return model_path
    
    def load_model(self, version: str) -> None:
        """
        Load a trained model
        
        Args:
            version (str): Model version to load
        """
        model_path = os.path.join('models', self.problem_type, self.model_name, f'{version}.joblib')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.best_model = joblib.load(model_path)
        
        # Load metrics
        metrics_path = os.path.join('models', self.problem_type, self.model_name, f'{version}_metrics.json')
        with open(metrics_path, 'r') as f:
            self.metrics = json.load(f)
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Predictions
        """
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        return self.best_model.predict(data) 