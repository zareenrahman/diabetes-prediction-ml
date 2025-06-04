# diabetes-prediction-ml
# A Machine Learning Model to Predict Diabetes from Clinical Data
This is a simple machine learning project where I built a model to predict whether a person is likely to have diabetes based on some medical measurements.

## Dataset
- Source: [Kaggle - Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 9 clinical attributes including glucose, BMI, insulin, age, and more.

## Files
- `diabetes_model.ipynb`: Full notebook with code, evaluation, and plots
- `diabetes.csv`: Input dataset
- `confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`: Output images

## What’s in this project
- A Jupyter notebook that walks through the full process: loading data, cleaning it, training a model, and evaluating the results.
- A Random Forest classifier trained on 9 features.
- Plots showing the model’s performance, including the confusion matrix, ROC curve, and feature importance.
- Final model achieves around **75% accuracy** and an **AUC of 0.83**.

## Tools Used
- Python, Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- scikit-learn

## How to use it
If you want to try this out yourself:
1. Clone this repo
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
3. Open the notebook in Jupyter and run through the cells:
   Jupyter Notebook diabetes_model.ipynb
