# Hyperparameter Tuning Comparison: Grid Search vs Random Search vs Bayesian Optimisation

**Student Name:** Pavan Kalyan Madhagoni  
**Student ID:** 24082699  
**Course:** Machine Learning Neural Networks  
**Assignment:** Individual Assignment (40%)

## 📋 Overview

This project compares three hyperparameter tuning strategies for a Support Vector Machine (SVM) classifier on the Wine Quality dataset:

- **Grid Search** - Exhaustive search on a fixed hyperparameter grid
- **Random Search** - Random sampling from hyperparameter distributions
- **Bayesian Optimisation** - Guided search using a surrogate model (Gaussian Process)

The goal is to understand how each method explores the hyperparameter space, their performance, and efficiency in using the evaluation budget.

## 🎯 Objectives

1. Implement and compare three hyperparameter tuning methods
2. Evaluate performance using F1-score (suitable for slightly imbalanced binary classification)
3. Analyze the trade-offs between exploration efficiency and final model performance
4. Compare cross-validation performance vs test set generalization

## 📊 Dataset

**Wine Quality Red Dataset** from UCI Machine Learning Repository

- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Direct Download:** [winequality-red.csv](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)
- **Size:** 1,599 samples with 12 features
- **Task:** Binary classification (quality ≥ 6 → "good", quality ≤ 5 → "not good")
- **Features:** Physicochemical measurements (acidity, sugar, sulphates, pH, etc.)

The dataset is automatically downloaded when running the notebook if not present locally.

## 🛠️ Technologies Used

- **Python 3.12+**
- **scikit-learn** - Machine learning models and evaluation
- **scikit-optimize** - Bayesian optimization (optional)
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **matplotlib & seaborn** - Data visualization
- **scipy** - Statistical distributions

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Optional: Bayesian Optimisation

The notebook includes Bayesian optimisation using `scikit-optimize`. If you want to use this feature:

```bash
pip install scikit-optimize
```

If `scikit-optimize` is not installed, the notebook will automatically skip Bayesian optimisation and only run Grid Search and Random Search.

## 🚀 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook "Coding notebook.ipynb"
```

2. Run all cells sequentially. The notebook will:
   - Automatically download the dataset if needed
   - Perform data preprocessing
   - Run all three hyperparameter tuning methods
   - Generate comparison visualizations
   - Evaluate models on test set

3. View results:
   - Cross-validation F1-scores for each method
   - Test set performance metrics
   - Confusion matrices
   - Comparison visualizations

## 📈 Methodology

### 1. Data Preprocessing
- Load Wine Quality Red dataset
- Convert quality scores to binary labels (good/not good)
- Train-test split (80/20) with stratification
- Feature scaling using StandardScaler

### 2. Model Pipeline
- **Base Model:** SVM with RBF kernel
- **Pipeline:** StandardScaler → SVC
- **Evaluation Metric:** F1-score (5-fold cross-validation)

### 3. Hyperparameter Spaces

**Grid Search:**
- C: [0.1, 1, 10, 100]
- gamma: [0.001, 0.01, 0.1, 1.0]
- Total combinations: 16

**Random Search:**
- C: log-uniform(1e-2, 1e2)
- gamma: log-uniform(1e-3, 1e1)
- Iterations: 40

**Bayesian Optimisation:**
- C: Real(1e-2, 1e2, prior="log-uniform")
- gamma: Real(1e-3, 1e1, prior="log-uniform")
- Iterations: 30

### 4. Evaluation
- Cross-validation F1-score for hyperparameter selection
- Test set F1-score for final model evaluation
- Classification reports and confusion matrices

## 📊 Results Summary

The notebook generates comprehensive comparisons including:

1. **Cross-Validation Performance:** Best mean CV F1-score for each method
2. **Test Set Performance:** Generalization performance on held-out test set
3. **Hyperparameter Values:** Optimal C and gamma for each method
4. **Visualizations:**
   - Class distribution plot
   - CV F1-score comparison bar chart
   - CV vs Test F1-score comparison

### Key Findings

- All three methods achieve similar performance (~0.78 F1-score)
- Grid Search: Simple and exhaustive but limited to predefined grid
- Random Search: Explores wider ranges efficiently
- Bayesian Optimisation: Guided search that learns from previous evaluations

## 📁 Project Structure

```
.
├── Coding notebook.ipynb          # Main Jupyter notebook
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── winequality-red.csv            # Dataset (auto-downloaded)
```

## 🔧 Configuration

The notebook uses a fixed random state for reproducibility:
- **RANDOM_STATE:** 24082699 (Student ID)

All random operations (data splitting, random search, etc.) use this seed to ensure reproducible results.

## 📝 Key Features

- ✅ Automatic dataset download
- ✅ Comprehensive hyperparameter tuning comparison
- ✅ Detailed performance metrics and visualizations
- ✅ Reproducible results with fixed random seeds
- ✅ Graceful handling of optional dependencies
- ✅ Well-documented code with markdown explanations

## 🎓 Learning Outcomes

This project demonstrates:

1. **Hyperparameter Tuning Methods:** Understanding differences between grid, random, and Bayesian search
2. **Model Evaluation:** Proper use of cross-validation and test set evaluation
3. **Performance Metrics:** Choosing appropriate metrics (F1-score) for imbalanced classification
4. **Reproducibility:** Setting random seeds for consistent results
5. **Data Science Workflow:** Complete pipeline from data loading to model evaluation

## 📚 References

- UCI Machine Learning Repository: [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- scikit-learn Documentation: [Model Selection](https://scikit-learn.org/stable/modules/model_selection.html)
- scikit-optimize Documentation: [Bayesian Optimization](https://scikit-optimize.github.io/stable/)

## 👤 Author

**Pavan Kalyan Madhagoni**  
Student ID: 24082699

## 📄 License

This project is part of an academic assignment. Please use responsibly and cite appropriately.

---

**Note:** This notebook is designed for educational purposes to compare hyperparameter tuning strategies. Results may vary slightly depending on the environment and random seed initialization.

