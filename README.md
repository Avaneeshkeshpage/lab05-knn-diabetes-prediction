# Lab 5 – K-Nearest Neighbors (KNN) on the Diabetes Dataset

Machine Learning Laboratory | B.Tech AIML | Ramdeobaba University

## Aim
To implement the K-Nearest Neighbor (KNN) algorithm and evaluate it on a real-world dataset.

## Dataset
Pima Indians Diabetes Dataset: 768 records, 8 medical predictors
(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age) and a binary target `Outcome`
(1 = diabetic, 0 = non-diabetic).

## What the notebook does
- Exploratory data analysis: class distribution, histograms, box plots, correlations, pair plot
- Data cleaning: invalid zeros replaced with NaN, then median imputation
- Stratified 80:20 train-test split and StandardScaler feature scaling
- KNN classifier (K = 5), evaluated with accuracy, precision, recall, F1-score and a confusion matrix
- Effect of K (1–30) on training and testing accuracy
- Best K selected with 5-fold stratified cross-validation
- Hyper-parameter tuning with GridSearchCV (n_neighbors, weights, distance metric)

## Results
| Model | Result |
|---|---|
| KNN (K = 5), test accuracy | 75.3% |
| Diabetic class: precision / recall / F1 | 0.66 / 0.61 / 0.63 |
| Best K from cross-validation | 13 (CV accuracy 77.0%) |
| GridSearchCV best | K = 19, euclidean, uniform (CV accuracy 77.7%) |

## Tech stack
Python, NumPy, pandas, Matplotlib, Seaborn, scikit-learn (Google Colab)

## Files
- `Lab5_KNN.ipynb`: main notebook
- `diabetes.csv`: dataset

## Author
Avaneesh Keshpage | Section A | Roll No. 64 | Batch A4
