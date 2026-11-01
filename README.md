# P24 - Gym Rat Level Prediction Using KNN

A complete end-to-end machine learning project that predicts a gym member Experience Level using the K-Nearest Neighbours (KNN) classification algorithm. All data is embedded directly in the notebook. No CSV upload required. Runs on Google Colab and Jupyter Notebook.

---

## Problem Statement

Given a gym member's biometric and workout data, predict their fitness experience level.

| Label | Meaning      |
|-------|--------------|
| 1     | Beginner     |
| 2     | Intermediate |
| 3     | Advanced     |

---

## Dataset

The dataset contains 100 synthetic gym member records with the following features.

| Feature                        | Description                          |
|--------------------------------|--------------------------------------|
| Age                            | Member age in years                  |
| Gender                         | Male or Female                       |
| Weight (kg)                    | Body weight                          |
| Height (m)                     | Body height                          |
| Max_BPM                        | Maximum heart rate during workout    |
| Avg_BPM                        | Average heart rate during workout    |
| Resting_BPM                    | Resting heart rate                   |
| Session_Duration (hours)       | Length of workout session            |
| Calories_Burned                | Calories burned per session          |
| Workout_Type                   | Cardio, Yoga, Strength, or HIIT      |
| Fat_Percentage                 | Body fat percentage                  |
| Water_Intake (liters)          | Daily water intake                   |
| Workout_Frequency (days/week)  | Days trained per week                |
| Experience_Level               | Target variable: 1, 2, or 3         |

---

## Pipeline Overview

```
Raw Data
  -> Data Cleaning
  -> Exploratory Data Analysis
  -> Feature Engineering
  -> Train-Test Split (80/20 stratified)
  -> Feature Selection (SelectKBest, ANOVA F-test, top 15)
  -> Feature Scaling (StandardScaler)
  -> Baseline KNN (k=5, Euclidean, scaled)
  -> Hyperparameter Tuning (k, metric, scaling)
  -> Final Model (best by Macro F1)
  -> Sample Predictions
  -> Save Results to CSV
```

---

## Steps in Detail

1. Dataset - Data hardcoded in Python. No external file needed.
2. Data Cleaning - Remove duplicates. Fill missing numeric values with median, categorical with mode.
3. Exploratory Data Analysis - Class distribution bar chart, feature histograms, boxplots per class, correlation heatmap.
4. Feature Engineering - New feature Water_per_kg (water intake divided by body weight). One-hot encoding for categorical columns.
5. Train-Test Split - 80 percent train, 20 percent test. Stratified by target class.
6. Feature Selection - SelectKBest with ANOVA F-test. Selects top 15 features.
7. Feature Scaling - StandardScaler applied to training data, then applied to test data.
8. Model Evaluation - Reports Accuracy, Precision (macro), Recall (macro), F1 (macro), Specificity (per-class average), AUC-ROC (macro OvR).
9. Baseline KNN - k=5, Euclidean distance, scaled features. Confusion matrix and classification report printed.
10. Hyperparameter Tuning - Tests k values 1, 3, 5, 7, 9, 11, 13, 15. Tests Euclidean and Manhattan distance. Tests scaled and unscaled inputs.
11. Final Model - Best configuration selected automatically by Macro F1 score. Final confusion matrix and classification report printed.
12. Sample Predictions - Predicts experience level for 3 example inputs.
13. Save Results - All model comparison metrics saved to knn_model_comparison.csv.

---

## Metrics Reported

- Accuracy
- Precision (macro average)
- Recall (macro average)
- F1 Score (macro average)
- Specificity (per-class average)
- AUC-ROC (macro one-vs-rest)

---

## How to Run

### Google Colab

1. Open https://colab.research.google.com
2. Upload P24_Gym_Rat_Level_Prediction_KNN.ipynb
3. Click Runtime then Run All

### Jupyter Notebook (Local)

```bash
git clone https://github.com/akpatel2029-ux/gym-rat-level-prediction-knn.git
cd gym-rat-level-prediction-knn
pip install numpy pandas matplotlib seaborn scikit-learn notebook
jupyter notebook P24_Gym_Rat_Level_Prediction_KNN.ipynb
```

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
notebook
```

Install all:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn notebook
```

---

## Project Structure

```
gym-rat-level-prediction-knn/
    P24_Gym_Rat_Level_Prediction_KNN.ipynb
    knn_model_comparison.csv
    README.md
```

---

## Algorithm: K-Nearest Neighbours

KNN is a non-parametric, instance-based learning algorithm. It classifies a data point based on the majority class among its k nearest neighbours in feature space.

Key choices explored in this project:

- k value: controls the bias-variance tradeoff
- Distance metric: Euclidean vs Manhattan
- Feature scaling: KNN is sensitive to feature magnitudes, so StandardScaler is applied and compared against unscaled inputs

---

## Author

AK Patel
GitHub: https://github.com/akpatel2029-ux
