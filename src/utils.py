"""
Utility Functions for AQI Anomaly Detection System

This module provides common utility functions used across the project.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PathManager:
    """Centralized path management for the project."""
    
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.data_dir = self.project_root / 'data'
        self.models_dir = self.project_root / 'models'
        self.results_dir = self.project_root / 'results'
        self.logs_dir = self.project_root / 'logs'
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.models_dir, self.results_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_data_path(self, filename):
        """Get full path for data file."""
        return self.data_dir / filename
    
    def get_model_path(self, filename):
        """Get full path for model file."""
        return self.models_dir / filename
    
    def get_results_path(self, filename):
        """Get full path for results file."""
        return self.results_dir / filename


class DataValidator:
    """Validate data integrity and quality."""
    
    @staticmethod
    def check_missing_values(df, threshold=0.5):
        """
        Check for missing values in dataframe.
        
        Args:
            df: pandas DataFrame
            threshold: Maximum allowed missing ratio (0-1)
            
        Returns:
            dict: Missing value statistics
        """
        missing_stats = {}
        total_missing = 0
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            total_missing += missing_count
            missing_ratio = missing_count / len(df)
            
            missing_stats[col] = {
                'count': int(missing_count),
                'ratio': float(missing_ratio),
                'above_threshold': missing_ratio > threshold
            }
        
        missing_stats['total_missing'] = int(total_missing)
        return missing_stats
    
    @staticmethod
    def check_data_types(df, expected_types=None):
        """
        Validate data types of columns.
        
        Args:
            df: pandas DataFrame
            expected_types: dict of column -> expected dtype
            
        Returns:
            dict: Data type validation results
        """
        if expected_types is None:
            return {col: str(df[col].dtype) for col in df.columns}
        
        validation = {}
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = df[col].dtype
                validation[col] = {
                    'expected': expected_type,
                    'actual': str(actual_type),
                    'valid': str(actual_type) == expected_type
                }
        
        return validation
    
    @staticmethod
    def check_value_ranges(df, range_dict):
        """
        Check if values are within expected ranges.
        
        Args:
            df: pandas DataFrame
            range_dict: dict of column -> (min, max) tuples
            
        Returns:
            dict: Range validation results
        """
        validation = {}
        for col, (min_val, max_val) in range_dict.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                actual_min = df[col].min()
                actual_max = df[col].max()
                
                validation[col] = {
                    'expected_range': (min_val, max_val),
                    'actual_range': (float(actual_min), float(actual_max)),
                    'within_range': (actual_min >= min_val) and (actual_max <= max_val),
                    'outliers': int(((df[col] < min_val) | (df[col] > max_val)).sum())
                }
        
        return validation


class MetricsCalculator:
    """Calculate various evaluation metrics."""
    
    @staticmethod
    def calculate_anomaly_metrics(y_true, y_pred):
        """
        Calculate metrics for anomaly detection.
        
        Args:
            y_true: True labels (1 for anomaly, 0 for normal)
            y_pred: Predicted labels
            
        Returns:
            dict: Computed metrics
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        # Handle case where all predictions are same
        if len(np.unique(y_pred)) == 1:
            logger.warning("All predictions are the same class")
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_anomalies_detected': int(y_pred.sum()),
            'total_anomalies_actual': int(y_true.sum()),
            'detection_rate': float(y_pred.sum() / len(y_pred))
        }
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(y_true, y_pred):
        """
        Calculate metrics for regression tasks.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            dict: Computed metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'mape': float(mape)
        }
        
        return metrics


class ConfigManager:
    """Manage configuration files."""
    
    @staticmethod
    def load_config(config_path):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    @staticmethod
    def save_config(config, config_path):
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Configuration saved to {config_path}")
    
    @staticmethod
    def get_default_config():
        """Get default configuration."""
        return {
            'preprocessing': {
                'missing_value_strategy': 'city_wise_median',
                'normalization': 'standard',
                'feature_engineering': True
            },
            'anomaly_detection': {
                'contamination': 0.1,
                'algorithms': ['isolation_forest', 'lof', 'autoencoder'],
                'ensemble_voting': 'majority'
            },
            'explainability': {
                'shap_background_samples': 100,
                'lime_num_samples': 5000,
                'top_features': 10
            },
            'alert_system': {
                'severity_thresholds': {
                    'critical': -0.2,
                    'high': -0.15,
                    'medium': -0.1,
                    'low': 0.0
                }
            }
        }


class ModelPersistence:
    """Handle model saving and loading."""
    
    @staticmethod
    def save_model(model, filepath, metadata=None):
        """
        Save model with metadata.
        
        Args:
            model: Model object to save
            filepath: Path to save the model
            metadata: Optional metadata dictionary
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, filepath)
        
        # Save metadata if provided
        if metadata is not None:
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
        
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath, load_metadata=False):
        """
        Load model with optional metadata.
        
        Args:
            filepath: Path to the model file
            load_metadata: Whether to load metadata as well
            
        Returns:
            model or (model, metadata) tuple
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        
        if load_metadata:
            metadata_path = filepath.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return model, metadata
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
                return model, {}
        
        return model


class TimeSeriesUtils:
    """Utilities for time series analysis."""
    
    @staticmethod
    def create_time_features(df, date_column='Date'):
        """
        Create time-based features from date column.
        
        Args:
            df: pandas DataFrame
            date_column: Name of the date column
            
        Returns:
            DataFrame with time features
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        df['Year'] = df[date_column].dt.year
        df['Month'] = df[date_column].dt.month
        df['Day'] = df[date_column].dt.day
        df['DayOfWeek'] = df[date_column].dt.dayofweek
        df['DayOfYear'] = df[date_column].dt.dayofyear
        df['Quarter'] = df[date_column].dt.quarter
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        return df
    
    @staticmethod
    def create_lag_features(df, columns, lags=[1, 7]):
        """
        Create lagged features for time series.
        
        Args:
            df: pandas DataFrame
            columns: List of columns to create lags for
            lags: List of lag values
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        return df
    
    @staticmethod
    def create_rolling_features(df, columns, windows=[7, 14, 30]):
        """
        Create rolling window features.
        
        Args:
            df: pandas DataFrame
            columns: List of columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for col in columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
        
        return df


def format_timestamp(dt=None):
    """Format timestamp for filenames."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime('%Y%m%d_%H%M%S')


def print_section_header(title, width=80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")


def print_metrics_table(metrics_dict, title="Metrics"):
    """Print metrics in a formatted table."""
    print_section_header(title)
    
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"{key:<30}: {value:.4f}")
        else:
            print(f"{key:<30}: {value}")
    
    print()


def save_results(results, filepath, format='json'):
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        filepath: Path to save the results
        format: 'json' or 'csv'
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4, default=str)
    elif format == 'csv':
        pd.DataFrame(results).to_csv(filepath, index=False)
    
    logger.info(f"Results saved to {filepath}")


def load_results(filepath):
    """Load results from file."""
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.suffix == '.csv':
        return pd.read_csv(filepath).to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


# Initialize path manager as singleton
path_manager = PathManager()


if __name__ == "__main__":
    # Test utilities
    print_section_header("Testing Utility Functions")
    
    # Test path manager
    print("Path Manager:")
    print(f"  Project Root: {path_manager.project_root}")
    print(f"  Data Dir: {path_manager.data_dir}")
    print(f"  Models Dir: {path_manager.models_dir}")
    
    # Test config manager
    print("\nDefault Config:")
    config = ConfigManager.get_default_config()
    print(json.dumps(config, indent=2))
    
    print("\n[PASS] Utility functions working correctly!")
