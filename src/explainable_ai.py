"""
Explainable AI Module for AQI Anomaly Detection

Implements SHAP (SHapley Additive exPlanations) and LIME to explain why certain
air quality readings are flagged as anomalous.

This module provides:
1. SHAP value calculation for global and local explanations
2. LIME explanations for individual anomalies
3. Feature importance ranking
4. Visual explanations (waterfall, force, summary plots)

Author: TY Sem 5 AIML Student
Date: November 2025
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Explainable AI libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("[WARNING] LIME not available. Install with: pip install lime")

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

RESULTS_DIR.mkdir(exist_ok=True)


class SHAPExplainer:
    """SHAP-based explainability for anomaly detection"""
    
    def __init__(self, model, X_background, feature_names):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained anomaly detection model
            X_background: Background dataset for SHAP (typically training data sample)
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self):
        """Create SHAP explainer"""
        print("\nCreating SHAP Explainer...")
        
        # Use KernelExplainer for model-agnostic explanations
        # This works with any black-box model
        def model_predict(X):
            """Wrapper function for model prediction"""
            try:
                # For isolation forest and LOF, use decision_function or score_samples
                if hasattr(self.model, 'decision_function'):
                    return self.model.decision_function(X)
                elif hasattr(self.model, 'score_samples'):
                    return self.model.score_samples(X)
                else:
                    return self.model.predict(X)
            except:
                return np.zeros(len(X))
        
        # Sample background data for faster computation
        background_sample = shap.sample(self.X_background, min(100, len(self.X_background)))
        
        self.explainer = shap.KernelExplainer(
            model_predict,
            background_sample,
            link="identity"
        )
        
        print("[SUCCESS] SHAP Explainer created successfully!")
    
    def explain_anomalies(self, X_anomalies, max_samples=50):
        """
        Generate SHAP explanations for anomalous instances
        
        Args:
            X_anomalies: Anomalous instances to explain
            max_samples: Maximum number of samples to explain (for performance)
        """
        print(f"\nGenerating SHAP explanations for {len(X_anomalies)} anomalies...")
        
        # Limit samples for performance
        if len(X_anomalies) > max_samples:
            print(f"   Sampling {max_samples} anomalies for explanation...")
            sample_indices = np.random.choice(len(X_anomalies), max_samples, replace=False)
            X_sample = X_anomalies[sample_indices]
        else:
            X_sample = X_anomalies
        
        # Calculate SHAP values
        print("   Computing SHAP values (this may take a few minutes)...")
        self.shap_values = self.explainer.shap_values(X_sample)
        
        print(f"[SUCCESS] SHAP values computed for {len(X_sample)} instances")
        
        return self.shap_values
    
    def plot_summary(self, save_path=None):
        """Create SHAP summary plot"""
        if self.shap_values is None:
            print("[WARNING] No SHAP values available. Run explain_anomalies() first.")
            return
        
        print("\nCreating SHAP summary plot...")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            features=self.X_background[:len(self.shap_values)],
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        plt.title("SHAP Feature Importance for Anomalies", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SUCCESS] Summary plot saved to {save_path}")
        
        plt.close()
    
    def plot_waterfall(self, instance_idx=0, save_path=None):
        """Create SHAP waterfall plot for a single instance"""
        if self.shap_values is None:
            print("[WARNING] No SHAP values available. Run explain_anomalies() first.")
            return
        
        print(f"\n💧 Creating SHAP waterfall plot for instance {instance_idx}...")
        
        # Create explanation object
        explanation = shap.Explanation(
            values=self.shap_values[instance_idx],
            base_values=self.explainer.expected_value,
            data=self.X_background[instance_idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.title(f"SHAP Waterfall Plot - Anomaly {instance_idx}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SUCCESS] Waterfall plot saved to {save_path}")
        
        plt.close()
    
    def get_feature_importance(self):
        """Get feature importance ranking from SHAP values"""
        if self.shap_values is None:
            print("[WARNING] No SHAP values available. Run explain_anomalies() first.")
            return None
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': mean_abs_shap
        }).sort_values('Importance', ascending=False)
        
        return importance_df


class LIMEExplainer:
    """LIME-based explainability for anomaly detection"""
    
    def __init__(self, model, X_train, feature_names):
        """
        Initialize LIME explainer
        
        Args:
            model: Trained anomaly detection model
            X_train: Training dataset
            feature_names: List of feature names
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required. Install with: pip install lime")
        
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
    
    def create_explainer(self):
        """Create LIME explainer"""
        print("\nCreating LIME Explainer...")
        
        # Prediction function for LIME
        def predict_fn(X):
            """Wrapper for model prediction"""
            try:
                if hasattr(self.model, 'decision_function'):
                    scores = self.model.decision_function(X)
                elif hasattr(self.model, 'score_samples'):
                    scores = self.model.score_samples(X)
                else:
                    scores = self.model.predict(X)
                
                # Convert to probabilities (0 = normal, 1 = anomaly)
                # Normalize scores to [0, 1] range
                scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                
                # Return as 2-class probabilities
                return np.column_stack([1 - scores_norm, scores_norm])
            except:
                return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])
        
        # Create LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=self.feature_names,
            class_names=['Normal', 'Anomaly'],
            mode='classification',
            discretize_continuous=True
        )
        
        print("[SUCCESS] LIME Explainer created successfully!")
    
    def explain_instance(self, instance, num_features=10):
        """
        Explain a single anomalous instance
        
        Args:
            instance: Single data instance to explain
            num_features: Number of top features to include in explanation
        """
        # Prediction function
        def predict_fn(X):
            try:
                if hasattr(self.model, 'decision_function'):
                    scores = self.model.decision_function(X)
                elif hasattr(self.model, 'score_samples'):
                    scores = self.model.score_samples(X)
                else:
                    scores = self.model.predict(X)
                
                scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                return np.column_stack([1 - scores_norm, scores_norm])
            except:
                return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features
        )
        
        return explanation
    
    def visualize_explanation(self, explanation, save_path=None):
        """Visualize LIME explanation"""
        print("\nCreating LIME explanation visualization...")
        
        fig = explanation.as_pyplot_figure()
        plt.title("LIME Feature Contributions", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SUCCESS] LIME visualization saved to {save_path}")
        
        plt.close()


class ExplainableAnomalyDetection:
    """Main class combining anomaly detection with explainability"""
    
    def __init__(self):
        """Initialize the explainable system"""
        self.X_train = None
        self.X_test = None
        self.feature_names = None
        self.anomaly_indices = None
        self.models = {}
        self.shap_explainers = {}
        self.lime_explainers = {}
    
    def load_data_and_models(self):
        """Load preprocessed data and trained models"""
        print("=" * 80)
        print("LOADING DATA AND MODELS")
        print("=" * 80)
        
        try:
            # Load features
            features_path = DATA_DIR / 'features.csv'
            X = pd.read_csv(features_path).values
            
            # Load feature names
            feature_cols_path = MODELS_DIR / 'feature_columns.pkl'
            self.feature_names = joblib.load(feature_cols_path)
            
            # Split data (same split as training)
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test = train_test_split(
                X, test_size=0.2, random_state=42
            )
            
            print(f"[SUCCESS] Data loaded: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test")
            
            # Load models
            model_files = {
                'isolation_forest': MODELS_DIR / 'isolation_forest_model.pkl',
                'lof': MODELS_DIR / 'lof_model.pkl'
            }
            
            for name, path in model_files.items():
                if path.exists():
                    self.models[name] = joblib.load(path)
                    print(f"[SUCCESS] Loaded {name} model")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading data/models: {str(e)}")
            return False
    
    def detect_anomalies(self):
        """Detect anomalies using loaded models"""
        print("\n" + "=" * 80)
        print("DETECTING ANOMALIES")
        print("=" * 80)
        
        # Use first available model for detection
        if not self.models:
            print("[ERROR] No models available")
            return False
        
        model_name = list(self.models.keys())[0]
        model = self.models[model_name]
        
        print(f"Using {model_name} for anomaly detection...")
        
        # Predict
        predictions = model.predict(self.X_test)
        
        # Find anomaly indices
        self.anomaly_indices = np.where(predictions == -1)[0]
        
        print(f"[SUCCESS] Detected {len(self.anomaly_indices)} anomalies ({len(self.anomaly_indices)/len(self.X_test)*100:.2f}%)")
        
        return True
    
    def create_explainers(self):
        """Create SHAP and LIME explainers"""
        print("\n" + "=" * 80)
        print("CREATING EXPLAINERS")
        print("=" * 80)
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            
            # SHAP explainer
            if SHAP_AVAILABLE:
                try:
                    self.shap_explainers[name] = SHAPExplainer(
                        model=model.model if hasattr(model, 'model') else model,
                        X_background=self.X_train,
                        feature_names=self.feature_names
                    )
                    self.shap_explainers[name].create_explainer()
                except Exception as e:
                    print(f"   ⚠️  SHAP explainer creation failed: {str(e)}")
            
            # LIME explainer
            if LIME_AVAILABLE:
                try:
                    self.lime_explainers[name] = LIMEExplainer(
                        model=model.model if hasattr(model, 'model') else model,
                        X_train=self.X_train,
                        feature_names=self.feature_names
                    )
                    self.lime_explainers[name].create_explainer()
                except Exception as e:
                    print(f"   ⚠️  LIME explainer creation failed: {str(e)}")
    
    def generate_explanations(self):
        """Generate explanations for detected anomalies"""
        print("\n" + "=" * 80)
        print("📝 GENERATING EXPLANATIONS")
        print("=" * 80)
        
        if len(self.anomaly_indices) == 0:
            print("⚠️  No anomalies to explain")
            return
        
        # Get anomalous samples
        X_anomalies = self.X_test[self.anomaly_indices]
        
        # Generate SHAP explanations
        for name, explainer in self.shap_explainers.items():
            print(f"\n{name} - SHAP Analysis:")
            
            # Explain anomalies
            explainer.explain_anomalies(X_anomalies, max_samples=50)
            
            # Generate visualizations
            summary_path = RESULTS_DIR / f'shap_summary_{name}.png'
            explainer.plot_summary(save_path=summary_path)
            
            # Feature importance
            importance = explainer.get_feature_importance()
            if importance is not None:
                print("\n   Top 10 Most Important Features:")
                print(importance.head(10).to_string(index=False))
                
                # Save to CSV
                importance_path = RESULTS_DIR / f'feature_importance_{name}.csv'
                importance.to_csv(importance_path, index=False)
                print(f"\n   ✅ Feature importance saved to {importance_path}")
        
        # Generate LIME explanations (for first few anomalies)
        for name, explainer in self.lime_explainers.items():
            print(f"\n{name} - LIME Analysis:")
            print("   Explaining first 3 anomalies...")
            
            for i in range(min(3, len(X_anomalies))):
                explanation = explainer.explain_instance(X_anomalies[i], num_features=10)
                lime_path = RESULTS_DIR / f'lime_explanation_{name}_instance_{i}.png'
                explainer.visualize_explanation(explanation, save_path=lime_path)
    
    def run_full_pipeline(self):
        """Execute complete explainability pipeline"""
        print("\n" + "🚀" * 40)
        print("STARTING EXPLAINABLE AI PIPELINE")
        print("🚀" * 40)
        
        # Load data and models
        if not self.load_data_and_models():
            return False
        
        # Detect anomalies
        if not self.detect_anomalies():
            return False
        
        # Create explainers
        self.create_explainers()
        
        # Generate explanations
        self.generate_explanations()
        
        print("\n" + "=" * 80)
        print("✅ EXPLAINABLE AI PIPELINE COMPLETED!")
        print("=" * 80)
        print(f"\n📊 Results saved to: {RESULTS_DIR}")
        print("\n📝 Generated files:")
        print("   - SHAP summary plots")
        print("   - Feature importance rankings")
        print("   - LIME explanation visualizations")
        
        return True


def main():
    """Main execution function"""
    # Initialize system
    system = ExplainableAnomalyDetection()
    
    # Run pipeline
    success = system.run_full_pipeline()
    
    if success:
        print("\n✨ Explainable AI analysis complete!")
        print("📝 Next step: Run dashboard/streamlit_app.py to explore results")
    else:
        print("\n❌ Pipeline failed. Ensure anomaly_detectors.py was run first.")


if __name__ == "__main__":
    main()
