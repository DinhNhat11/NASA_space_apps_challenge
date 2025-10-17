# 🚀 Exoplanet Detection ML Model

An ensemble machine learning system for classifying Kepler exoplanet candidates using NASA's publicly available datasets.

**Target Audience**: Researchers  
**Models**: Random Forest + Gradient Boosting + Neural Network (Ensemble)

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Model Architecture](#model-architecture)
- [Output Files](#output-files)
- [Understanding Results](#understanding-results)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_exoplanet_model.py
```

### 3. Make Predictions

```bash
python predict_exoplanets.py your_data.csv
```

---

## 📦 Requirements

### Python Version
- Python 3.7 or higher

### Required Libraries
```
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
joblib >= 1.0.0
```

### Data Requirements

Your CSV file must contain these columns:
- `koi_period` - Orbital period (days)
- `koi_duration` - Transit duration (hours)
- `koi_depth` - Transit depth (ppm)
- `koi_prad` - Planetary radius (Earth radii)
- `koi_impact` - Impact parameter
- `koi_model_snr` - Signal-to-noise ratio
- `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec` - False positive flags
- `koi_steff`, `koi_slogg`, `koi_srad` - Stellar properties
- And other Kepler mission features

---

## 🎯 Training the Model

### Step 1: Prepare Your Data

Place your Kepler cumulative dataset CSV file in the same directory:
```
cumulative_2025.10.04_13.39.13.csv
```

### Step 2: Run Training Script

```bash
python train_exoplanet_model.py
```

### What Happens During Training:

1. **Data Loading**: Reads CSV (handles comment lines starting with #)
2. **Feature Engineering**: Creates 8 new features from physical relationships
3. **Preprocessing**: Handles missing values, scales features, clips outliers
4. **Model Training**: Trains 3 models (Random Forest, Gradient Boosting, Neural Network)
5. **Evaluation**: Tests on 20% holdout set
6. **Visualization**: Generates plots and saves model

### Training Output:

```
output/
├── exoplanet_classifier.pkl      # Trained model
├── confusion_matrix.png           # Performance visualization
├── feature_importance.png         # Top features
├── class_distribution.png         # Dataset overview
└── feature_importance.csv         # Feature rankings
```

### Expected Training Time:
- **~2-5 minutes** on a modern laptop (depends on CPU)

---

## 🔮 Making Predictions

### For New Data:

```bash
python predict_exoplanets.py new_candidates.csv
```

### Example with Test Data:

```bash
# Use a subset of your original data for testing
python predict_exoplanets.py test_data.csv
```

### Prediction Output:

The script generates:

1. **`your_file_predictions.csv`** - Full results with predictions and probabilities
2. **`your_file_summary.txt`** - Text summary of results

### Output Columns Added:

- `prediction` - Predicted class (CONFIRMED, CANDIDATE, FALSE POSITIVE)
- `confidence` - Model confidence (0-1)
- `prob_candidate` - Probability of being a candidate
- `prob_confirmed` - Probability of being confirmed
- `prob_false_positive` - Probability of being false positive

---

## 🏗️ Model Architecture

### Ensemble Approach (Soft Voting)

The model uses **three complementary algorithms**:

#### 1. **Random Forest** (200 trees)
- **Purpose**: Feature importance analysis
- **Strength**: Handles non-linear relationships
- **Config**: max_depth=15, balanced class weights

#### 2. **Gradient Boosting** (150 estimators)
- **Purpose**: High accuracy through error correction
- **Strength**: Sequential learning
- **Config**: learning_rate=0.1, max_depth=5

#### 3. **Neural Network** (100→50 neurons)
- **Purpose**: Complex pattern recognition
- **Strength**: Non-linear interactions
- **Config**: 2 hidden layers, ReLU activation, adaptive learning

### Prediction Method:
- Averages probabilities from all 3 models (soft voting)
- Takes class with highest average probability
- Provides confidence score (max probability)

---

## 📊 Understanding Results

### Classification Metrics

**Precision**: Of predicted planets, what % are actually planets?  
**Recall**: Of actual planets, what % did we find?  
**F1-Score**: Harmonic mean of precision and recall

### Confidence Levels

- **High (>90%)**: Strong prediction, high confidence
- **Medium (70-90%)**: Good prediction, moderate confidence
- **Low (<70%)**: Uncertain prediction, needs review

### Feature Importance

Top features typically include:
1. `koi_score` - Disposition score (most important)
2. `fp_flag_score` - Combined false positive flags
3. `koi_fpflag_ss` - Stellar eclipse flag
4. `koi_model_snr` - Signal-to-noise ratio
5. Physical parameters (period, depth, radius)

---

## 🔬 For Researchers

### Model Interpretability

```python
import joblib
import pandas as pd

# Load model
classifier = joblib.load('output/exoplanet_classifier.pkl')

# Check feature importance
print(classifier.feature_importance)

# Access individual models
rf_model = classifier.models['random_forest']
gb_model = classifier.models['gradient_boosting']
nn_model = classifier.models['neural_network']

# Get detailed probabilities
predictions, confidence, probabilities = classifier.predict_ensemble(X_test)
```

### Customization

To modify model parameters, edit `train_exoplanet_model.py`:

```python
# Example: Change Random Forest depth
self.models['random_forest'] = RandomForestClassifier(
    n_estimators=300,  # More trees
    max_depth=20,      # Deeper trees
    # ... other parameters
)
```

### Cross-Validation

For more robust evaluation, add cross-validation:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
print(f"CV F1-Score: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## 🐛 Troubleshooting

### Issue: "Data file not found"
**Solution**: Ensure CSV file is in the same directory as the script

### Issue: "Missing required features"
**Solution**: Check that your CSV has all required columns (see Data Requirements)

### Issue: "Model file not found" (when predicting)
**Solution**: Run training first: `python train_exoplanet_model.py`

### Issue: Poor performance
**Solutions**:
1. Check data quality (missing values, outliers)
2. Ensure class balance isn't too extreme
3. Try hyperparameter tuning
4. Increase training data size

---

## 📈 Performance Expectations

### Typical Results (Kepler Dataset):

- **Overall Accuracy**: ~95-98%
- **Confirmed Planets**: Precision ~90-95%, Recall ~85-90%
- **False Positives**: Precision ~96-98%, Recall ~95-97%
- **Candidates**: Precision ~80-85%, Recall ~75-85%

*Note: Performance varies with data quality and class distribution*

---

## 📚 References

- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
- Kepler Mission: https://www.nasa.gov/kepler
- Scikit-learn Documentation: https://scikit-learn.org/


For issues or questions:
- Check the troubleshooting section
- Review error messages carefully
- Verify data format and requirements

---

**Happy Planet Hunting! 🌟🔭**
