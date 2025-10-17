"""
EXOPLANET CLASSIFICATION - TRAINING SCRIPT
==========================================
Run this script to train the ensemble model on Kepler data.

Usage:
    python train_exoplanet_model.py

Requirements:
    pip install numpy pandas scikit-learn matplotlib seaborn joblib
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# EXOPLANET CLASSIFIER CLASS
# ============================================================================

class ExoplanetClassifier:
    """Ensemble classifier for exoplanet detection using Kepler data."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        self.feature_names = None
        self.feature_importance = None
        
    def engineer_features(self, df):
        """Create engineered features based on astronomical knowledge."""
        df = df.copy()
        
        # Physical ratios
        df['period_duration_ratio'] = df['koi_period'] / (df['koi_duration'] + 1e-6)
        df['depth_radius_ratio'] = df['koi_depth'] / (df['koi_prad'] + 1e-6)
        df['impact_snr_product'] = df['koi_impact'] * df['koi_model_snr']
        
        # Aggregate FP flag score (weighted by importance)
        df['fp_flag_score'] = (
            df['koi_fpflag_ss'] * 0.46 +
            df['koi_fpflag_co'] * 0.39 +
            df['koi_fpflag_nt'] * 0.32 +
            df['koi_fpflag_ec'] * 0.24
        )
        
        # Log transforms for skewed distributions
        df['log_period'] = np.log1p(df['koi_period'])
        df['log_prad'] = np.log1p(df['koi_prad'])
        df['log_depth'] = np.log1p(df['koi_depth'])
        
        # Stellar-planet interaction
        df['stellar_planet_interaction'] = df['koi_srad'] * df['koi_prad']
        
        return df
    
    def select_features(self, df):
        """Select features for modeling."""
        base_features = [
            'koi_score',
            'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
            'fp_flag_score',
            'koi_period', 'koi_duration', 'koi_impact', 'log_period',
            'koi_depth', 'koi_model_snr', 'log_depth',
            'koi_prad', 'koi_teq', 'koi_insol', 'log_prad',
            'koi_steff', 'koi_slogg', 'koi_srad',
            'period_duration_ratio', 'depth_radius_ratio',
            'impact_snr_product', 'stellar_planet_interaction'
        ]
        
        available_features = [f for f in base_features if f in df.columns]
        return df[available_features]
    
    def preprocess_data(self, X, fit=True):
        """Preprocess features with robust handling."""
        X = X.copy()
        
        if fit:
            self.imputer = SimpleImputer(strategy='median')
            X_imputed = self.imputer.fit_transform(X)
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)
        
        X_clipped = np.clip(X_scaled, 
                           np.percentile(X_scaled, 1, axis=0), 
                           np.percentile(X_scaled, 99, axis=0))
        
        return X_clipped
    
    def build_models(self, X_train, y_train):
        """Build ensemble of complementary models."""
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"Class weights: {class_weight_dict}")
        
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=self.random_state
        )
        
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=self.random_state
        )
        
        print("\nTraining models...")
        for name, model in self.models.items():
            print(f"  ├─ Training {name}...", end=' ')
            model.fit(X_train, y_train)
            print("✓")
        
        if hasattr(self.models['random_forest'], 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models['random_forest'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        print("  └─ All models trained successfully!\n")
    
    def predict_ensemble(self, X):
        """Ensemble prediction using soft voting."""
        proba_predictions = []
        
        for model in self.models.values():
            proba = model.predict_proba(X)
            proba_predictions.append(proba)
        
        ensemble_proba = np.mean(proba_predictions, axis=0)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        confidence = np.max(ensemble_proba, axis=1)
        
        return ensemble_pred, confidence, ensemble_proba
    
    def fit(self, df, target_col='koi_disposition'):
        """Complete training pipeline."""
        print("="*70)
        print(" EXOPLANET CLASSIFIER - TRAINING PIPELINE")
        print("="*70)
        
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df[target_col])
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Total samples: {len(df)}")
        print(f"   Target classes: {list(self.label_encoder.classes_)}")
        for i, cls in enumerate(self.label_encoder.classes_):
            count = np.sum(y == i)
            pct = count / len(y) * 100
            print(f"   - {cls}: {count} ({pct:.1f}%)")
        
        print("\n🔧 Feature Engineering...")
        df_engineered = self.engineer_features(df)
        
        X = self.select_features(df_engineered)
        self.feature_names = X.columns.tolist()
        print(f"   Selected {len(self.feature_names)} features")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        print(f"\n📦 Data Split:")
        print(f"   Training set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        
        print(f"\n⚙️  Preprocessing...")
        X_train_processed = self.preprocess_data(X_train, fit=True)
        X_test_processed = self.preprocess_data(X_test, fit=False)
        
        print(f"\n🤖 Building Ensemble Models...")
        self.build_models(X_train_processed, y_train)
        
        print(f"📈 Evaluating Performance...")
        self.evaluate(X_test_processed, y_test)
        
        return self
    
    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation."""
        print("\n" + "="*70)
        print(" INDIVIDUAL MODEL PERFORMANCE")
        print("="*70)
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            print(f"\n{name.upper().replace('_', ' ')}:")
            print(classification_report(y_test, y_pred, 
                                       target_names=self.label_encoder.classes_,
                                       digits=3))
        
        print("\n" + "="*70)
        print(" ENSEMBLE MODEL PERFORMANCE")
        print("="*70)
        
        y_pred_ensemble, confidence, proba = self.predict_ensemble(X_test)
        
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred_ensemble,
                                   target_names=self.label_encoder.classes_,
                                   digits=3))
        
        print("\n📋 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_ensemble)
        cm_df = pd.DataFrame(cm, 
                            index=[f"True {c}" for c in self.label_encoder.classes_],
                            columns=[f"Pred {c}" for c in self.label_encoder.classes_])
        print(cm_df)
        
        print(f"\n🎯 Confidence Metrics:")
        print(f"   Average confidence: {np.mean(confidence):.3f}")
        print(f"   High confidence (>90%): {np.sum(confidence > 0.9) / len(confidence) * 100:.1f}%")
        print(f"   Medium confidence (70-90%): {np.sum((confidence > 0.7) & (confidence <= 0.9)) / len(confidence) * 100:.1f}%")
        print(f"   Low confidence (<70%): {np.sum(confidence <= 0.7) / len(confidence) * 100:.1f}%")
        
        if self.feature_importance is not None:
            print("\n" + "="*70)
            print(" TOP 15 MOST IMPORTANT FEATURES")
            print("="*70)
            print(self.feature_importance.head(15).to_string(index=False))
    
    def predict(self, df):
        """Predict on new data."""
        df_engineered = self.engineer_features(df)
        X = self.select_features(df_engineered)
        X_processed = self.preprocess_data(X, fit=False)
        
        predictions, confidence, proba = self.predict_ensemble(X_processed)
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        results = pd.DataFrame({
            'prediction': predicted_labels,
            'confidence': confidence,
        })
        
        # Add probability columns for each class
        for i, cls in enumerate(self.label_encoder.classes_):
            results[f'prob_{cls.lower().replace(" ", "_")}'] = proba[:, i]
        
        return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(cm, classes, save_path='confusion_matrix.png'):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Ensemble Model', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()

def plot_feature_importance(feature_importance, save_path='feature_importance.png'):
    """Plot top feature importances."""
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title('Top 15 Most Important Features', fontsize=16, pad=20)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved to {save_path}")
    plt.close()

def plot_class_distribution(y, classes, save_path='class_distribution.png'):
    """Plot class distribution."""
    plt.figure(figsize=(10, 6))
    counts = np.bincount(y)
    bars = plt.bar(classes, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Class Distribution in Dataset', fontsize=16, pad=20)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(y)*100:.1f}%)',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Class distribution plot saved to {save_path}")
    plt.close()


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print(" 🚀 EXOPLANET DETECTION MODEL - TRAINING")
    print("="*70)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Check if data file exists
    data_file = 'cumulative_2025.10.04_13.39.13.csv'
    if not os.path.exists(data_file):
        print(f"❌ ERROR: Data file '{data_file}' not found!")
        print(f"\n📁 Please ensure the CSV file is in the same directory as this script.")
        print(f"   Current directory: {os.getcwd()}")
        return
    
    # Load data
    print(f"📂 Loading data from {data_file}...")
    try:
        # Read CSV, skipping comment lines that start with #
        df = pd.read_csv(data_file, comment='#')
        print(f"✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns\n")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize and train classifier
    classifier = ExoplanetClassifier(random_state=42)
    classifier.fit(df, target_col='koi_disposition')
    
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the trained model
    model_path = os.path.join(output_dir, 'exoplanet_classifier.pkl')
    joblib.dump(classifier, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    
    # Re-run prediction on test set for visualizations
    df_engineered = classifier.engineer_features(df)
    X = classifier.select_features(df_engineered)
    y = classifier.label_encoder.transform(df['koi_disposition'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_test_processed = classifier.preprocess_data(X_test, fit=False)
    y_pred, _, _ = classifier.predict_ensemble(X_test_processed)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classifier.label_encoder.classes_, 
                         os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot feature importance
    if classifier.feature_importance is not None:
        plot_feature_importance(classifier.feature_importance,
                               os.path.join(output_dir, 'feature_importance.png'))
    
    # Plot class distribution
    plot_class_distribution(y, classifier.label_encoder.classes_,
                           os.path.join(output_dir, 'class_distribution.png'))
    
    # Save feature importance to CSV
    if classifier.feature_importance is not None:
        importance_path = os.path.join(output_dir, 'feature_importance.csv')
        classifier.feature_importance.to_csv(importance_path, index=False)
        print(f"✓ Feature importance saved to {importance_path}")
    
    print("\n" + "="*70)
    print(" ✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Output files saved in '{output_dir}/' directory:")
    print(f"   - exoplanet_classifier.pkl (trained model)")
    print(f"   - confusion_matrix.png")
    print(f"   - feature_importance.png")
    print(f"   - class_distribution.png")
    print(f"   - feature_importance.csv")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review the performance metrics above")
    print(f"   2. Check the visualizations in the output folder")
    print(f"   3. Use the saved model for predictions on new data")
    
    print(f"\n💡 To make predictions on new data:")
    print(f"   >>> import joblib")
    print(f"   >>> classifier = joblib.load('output/exoplanet_classifier.pkl')")
    print(f"   >>> predictions = classifier.predict(new_data_df)")
    
    print(f"\n Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
