import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

class MetricsBase:
    def __init__(self):
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.auc = None
        self.logloss = None

class AutoMLClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = MetricsBase()
    
    try:
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Empty prediction arrays")
            
        metrics.accuracy = accuracy_score(y_true, y_pred)
        metrics.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics.f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        if y_prob is not None:
            try:
                metrics.auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
                metrics.logloss = log_loss(y_true, y_prob)
            except Exception as e:
                logger.warning(f"Could not calculate AUC/LogLoss: {str(e)}")
                metrics.auc = 0.0
                metrics.logloss = 0.0
        else:
            metrics.auc = 0.0
            metrics.logloss = 0.0
            
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        metrics.accuracy = 0.0
        metrics.precision = 0.0
        metrics.recall = 0.0
        metrics.f1 = 0.0
        metrics.auc = 0.0
        metrics.logloss = 0.0
    
    return metrics

def preprocess_data(data):
    try:
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Identify numeric and categorical columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        # Handle numeric columns
        numeric_imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        
        if len(numeric_columns) > 0:
            df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        # Handle categorical columns
        le = LabelEncoder()
        for column in categorical_columns:
            try:
                df[column] = df[column].fillna('unknown')
                df[column] = le.fit_transform(df[column].astype(str))
            except Exception as e:
                logger.error(f"Error encoding column {column}: {str(e)}")
                df = df.drop(columns=[column])
        
        # Final check for any remaining non-numeric columns
        for column in df.columns:
            if not np.issubdtype(df[column].dtype, np.number):
                logger.warning(f"Dropping non-numeric column: {column}")
                df = df.drop(columns=[column])
        
        return df
        
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}")
        raise

def train_model(model, X_train, X_test, y_train, y_test):
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        try:
            y_prob = model.predict_proba(X_test)
        except (AttributeError, NotImplementedError):
            y_prob = None
            
        return calculate_metrics(y_test, y_pred, y_prob)
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return calculate_metrics([], [])

def train_all_models(data, target_column):
    try:
        # Preprocess the data
        processed_data = preprocess_data(data)
        
        if target_column not in processed_data.columns:
            raise ValueError(f"Target column '{target_column}' not found after preprocessing")
        
        X = processed_data.drop(columns=[target_column])
        y = processed_data[target_column]
        
        # Convert y to numeric if it's not already
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'GBM': GradientBoostingClassifier(random_state=42),
            'Deep Learning': MLPClassifier(random_state=42),
            'AutoML': AutoMLClassifier()
        }
        
        metrics = {}
        for name, model in models.items():
            try:
                logger.info(f"Training {name}")
                metrics[name] = train_model(model, X_train, X_test, y_train, y_test)
                logger.info(f"Completed {name}")
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                metrics[name] = calculate_metrics([], [])
        
        return models, metrics
        
    except Exception as e:
        logger.error(f"Error in train_all_models: {str(e)}")
        raise

def format_metrics_response(metrics):
    try:
        formatted_metrics = {}
        for model_name, model_metrics in metrics.items():
            formatted_metrics[model_name] = {
                'accuracy': float(model_metrics.accuracy),
                'precision': float(model_metrics.precision),
                'recall': float(model_metrics.recall),
                'f1': float(model_metrics.f1),
                'auc': float(model_metrics.auc),
                'logloss': float(model_metrics.logloss)
            }
        return formatted_metrics
    except Exception as e:
        logger.error(f"Error formatting metrics response: {str(e)}")
        raise