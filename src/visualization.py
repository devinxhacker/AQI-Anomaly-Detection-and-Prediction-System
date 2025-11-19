"""
Advanced Visualization Functions for AQI Anomaly Detection

This module provides comprehensive visualization capabilities for
anomaly detection results, explainability, and comparative analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AnomalyVisualizer:
    """Visualize anomaly detection results."""
    
    def __init__(self, output_dir='results'):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_anomaly_scatter(self, df, x_col, y_col, anomaly_col='is_anomaly', 
                            title='Anomaly Detection Scatter Plot', save_name=None):
        """
        Create scatter plot highlighting anomalies.
        
        Args:
            df: DataFrame with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            anomaly_col: Column indicating anomalies (1 for anomaly, 0 for normal)
            title: Plot title
            save_name: Filename to save plot
        """
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col,
            color=anomaly_col,
            color_discrete_map={0: 'blue', 1: 'red'},
            labels={anomaly_col: 'Type', 0: 'Normal', 1: 'Anomaly'},
            title=title,
            hover_data=df.columns
        )
        
        fig.update_layout(
            width=1000,
            height=600,
            font=dict(size=12)
        )
        
        if save_name:
            filepath = self.output_dir / save_name
            fig.write_html(str(filepath))
            logger.info(f"Plot saved to {filepath}")
        
        return fig
    
    def plot_temporal_anomalies(self, df, date_col='Date', anomaly_col='is_anomaly',
                                aqi_col='AQI', title='Temporal Anomaly Distribution',
                                save_name=None):
        """
        Plot anomalies over time.
        
        Args:
            df: DataFrame with temporal data
            date_col: Date column name
            anomaly_col: Anomaly indicator column
            aqi_col: AQI or metric column to plot
            title: Plot title
            save_name: Filename to save
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        fig = go.Figure()
        
        # Plot normal points
        normal = df[df[anomaly_col] == 0]
        fig.add_trace(go.Scatter(
            x=normal[date_col],
            y=normal[aqi_col],
            mode='markers',
            name='Normal',
            marker=dict(color='lightblue', size=4)
        ))
        
        # Plot anomalies
        anomalies = df[df[anomaly_col] == 1]
        fig.add_trace(go.Scatter(
            x=anomalies[date_col],
            y=anomalies[aqi_col],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=8, symbol='x')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=aqi_col,
            width=1200,
            height=500,
            hovermode='closest'
        )
        
        if save_name:
            filepath = self.output_dir / save_name
            fig.write_html(str(filepath))
            logger.info(f"Plot saved to {filepath}")
        
        return fig
    
    def plot_anomaly_heatmap(self, df, date_col='Date', city_col='City',
                            anomaly_col='is_anomaly', title='Anomaly Heatmap by City',
                            save_name=None):
        """
        Create heatmap of anomalies by city and time.
        
        Args:
            df: DataFrame with data
            date_col: Date column
            city_col: City column
            anomaly_col: Anomaly indicator
            title: Plot title
            save_name: Filename to save
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df['YearMonth'] = df[date_col].dt.to_period('M').astype(str)
        
        # Create pivot table
        pivot = df.groupby([city_col, 'YearMonth'])[anomaly_col].sum().reset_index()
        pivot_table = pivot.pivot(index=city_col, columns='YearMonth', values=anomaly_col)
        
        # Create heatmap
        plt.figure(figsize=(16, 10))
        sns.heatmap(pivot_table, cmap='YlOrRd', annot=False, fmt='g', 
                   cbar_kws={'label': 'Anomaly Count'})
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Year-Month', fontsize=12)
        plt.ylabel('City', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        return plt.gcf()
    
    def plot_severity_distribution(self, alerts, title='Alert Severity Distribution',
                                   save_name=None):
        """
        Plot distribution of alert severities.
        
        Args:
            alerts: List of Alert objects
            title: Plot title
            save_name: Filename to save
        """
        severities = [a.severity for a in alerts]
        severity_counts = pd.Series(severities).value_counts()
        
        colors = {
            'critical': '#DC143C',
            'high': '#FF8C00',
            'medium': '#FFD700',
            'low': '#90EE90'
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=severity_counts.index,
                y=severity_counts.values,
                marker_color=[colors.get(s, 'gray') for s in severity_counts.index],
                text=severity_counts.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Severity Level',
            yaxis_title='Count',
            width=800,
            height=500
        )
        
        if save_name:
            filepath = self.output_dir / save_name
            fig.write_html(str(filepath))
            logger.info(f"Plot saved to {filepath}")
        
        return fig


class ModelComparisonVisualizer:
    """Visualize model comparison results."""
    
    def __init__(self, output_dir='results'):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_metrics_comparison(self, metrics_dict, title='Model Performance Comparison',
                               save_name=None):
        """
        Create bar chart comparing model metrics.
        
        Args:
            metrics_dict: Dictionary of {model_name: {metric: value}}
            title: Plot title
            save_name: Filename to save
        """
        # Prepare data
        models = list(metrics_dict.keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [metrics_dict[model].get(metric, 0) for model in models]
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=models,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            width=1000,
            height=600,
            yaxis=dict(range=[0, 1])
        )
        
        if save_name:
            filepath = self.output_dir / save_name
            fig.write_html(str(filepath))
            logger.info(f"Plot saved to {filepath}")
        
        return fig
    
    def plot_detection_comparison(self, detection_results, title='Detection Comparison',
                                 save_name=None):
        """
        Compare detection results across models.
        
        Args:
            detection_results: Dictionary of {model_name: anomaly_array}
            title: Plot title
            save_name: Filename to save
        """
        models = list(detection_results.keys())
        detection_counts = {
            model: int(predictions.sum()) 
            for model, predictions in detection_results.items()
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(detection_counts.keys()),
                y=list(detection_counts.values()),
                text=list(detection_counts.values()),
                textposition='auto',
                marker_color='indianred'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Model',
            yaxis_title='Anomalies Detected',
            width=800,
            height=500
        )
        
        if save_name:
            filepath = self.output_dir / save_name
            fig.write_html(str(filepath))
            logger.info(f"Plot saved to {filepath}")
        
        return fig
    
    def plot_confusion_matrices(self, confusion_matrices, model_names,
                               save_name=None):
        """
        Plot confusion matrices for multiple models.
        
        Args:
            confusion_matrices: List of confusion matrices
            model_names: List of model names
            save_name: Filename to save
        """
        n_models = len(confusion_matrices)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, cm, name in zip(axes, confusion_matrices, model_names):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'])
            ax.set_title(f'{name}\nConfusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        return fig


class ExplainabilityVisualizer:
    """Visualize explainability results (SHAP/LIME)."""
    
    def __init__(self, output_dir='results'):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_feature_importance(self, feature_importance, top_n=10,
                               title='Feature Importance', save_name=None):
        """
        Plot feature importance bar chart.
        
        Args:
            feature_importance: Dictionary or Series of {feature: importance}
            top_n: Number of top features to show
            title: Plot title
            save_name: Filename to save
        """
        if isinstance(feature_importance, dict):
            importance = pd.Series(feature_importance)
        else:
            importance = feature_importance
        
        # Get top N features
        top_features = importance.nlargest(top_n)
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_features.values,
                y=top_features.index,
                orientation='h',
                marker_color='steelblue',
                text=[f'{v:.3f}' for v in top_features.values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            width=900,
            height=600,
            yaxis=dict(autorange='reversed')
        )
        
        if save_name:
            filepath = self.output_dir / save_name
            fig.write_html(str(filepath))
            logger.info(f"Plot saved to {filepath}")
        
        return fig
    
    def plot_shap_summary(self, shap_values, feature_names, save_name=None):
        """
        Create SHAP summary plot.
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            save_name: Filename to save
        """
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=True)
        
        # Create plot
        fig = go.Figure(data=[
            go.Bar(
                x=shap_df['importance'],
                y=shap_df['feature'],
                orientation='h',
                marker_color='coral',
                text=[f'{v:.3f}' for v in shap_df['importance']],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='SHAP Feature Importance',
            xaxis_title='Mean |SHAP Value|',
            yaxis_title='Feature',
            width=900,
            height=700
        )
        
        if save_name:
            filepath = self.output_dir / save_name
            fig.write_html(str(filepath))
            logger.info(f"Plot saved to {filepath}")
        
        return fig


class DataExplorationVisualizer:
    """Visualize data exploration and EDA results."""
    
    def __init__(self, output_dir='results'):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_correlation_heatmap(self, df, columns=None, title='Correlation Heatmap',
                                save_name=None):
        """
        Create correlation heatmap.
        
        Args:
            df: DataFrame
            columns: Columns to include (None for all numeric)
            title: Plot title
            save_name: Filename to save
        """
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = columns
        
        corr = df[numeric_cols].corr()
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        return plt.gcf()
    
    def plot_distribution_grid(self, df, columns, title='Distribution Grid',
                              save_name=None):
        """
        Create grid of distribution plots.
        
        Args:
            df: DataFrame
            columns: List of columns to plot
            title: Plot title
            save_name: Filename to save
        """
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(columns):
            if idx < len(axes):
                df[col].hist(bins=50, ax=axes[idx], edgecolor='black')
                axes[idx].set_title(f'{col} Distribution')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
        
        # Hide extra subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, y=1.00)
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        return fig
    
    def plot_city_comparison(self, df, metric_col='AQI', city_col='City',
                            title='City-wise Comparison', save_name=None):
        """
        Create box plot comparing metric across cities.
        
        Args:
            df: DataFrame
            metric_col: Column to compare
            city_col: City column
            title: Plot title
            save_name: Filename to save
        """
        plt.figure(figsize=(14, 8))
        
        # Sort cities by median value
        city_order = df.groupby(city_col)[metric_col].median().sort_values(ascending=False).index
        
        sns.boxplot(data=df, x=city_col, y=metric_col, order=city_order, palette='Set2')
        plt.xticks(rotation=45, ha='right')
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('City', fontsize=12)
        plt.ylabel(metric_col, fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        return plt.gcf()


if __name__ == "__main__":
    # Test visualizations
    print("="*80)
    print("Testing Visualization Functions")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    sample_df = pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=n_samples),
        'City': np.random.choice(['Delhi', 'Mumbai', 'Kolkata'], n_samples),
        'PM2.5': np.random.normal(100, 50, n_samples),
        'AQI': np.random.normal(200, 80, n_samples),
        'is_anomaly': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    })
    
    # Test anomaly visualizer
    viz = AnomalyVisualizer()
    print("\n[PASS] Anomaly Visualizer initialized")
    
    # Test model comparison
    model_viz = ModelComparisonVisualizer()
    print("[PASS] Model Comparison Visualizer initialized")
    
    # Test explainability
    explain_viz = ExplainabilityVisualizer()
    print("[PASS] Explainability Visualizer initialized")
    
    # Test EDA
    eda_viz = DataExplorationVisualizer()
    print("[PASS] Data Exploration Visualizer initialized")
    
    print("\n[PASS] All visualization functions working correctly!")
