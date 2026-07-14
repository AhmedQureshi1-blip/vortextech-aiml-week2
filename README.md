# vortextech-aiml-week2# vortextech-aiml-week2

Week 2 submission — Vortex Tech AI & ML Internship Track: **Build a Classification Model**.

## What this is

A Jupyter notebook (`week2_classification_model.ipynb`) that trains and evaluates a binary
classifier predicting Titanic passenger survival (`survived`: 0/1), using the dataset cleaned in
Week 1.

Steps covered:

- Loads `titanic_cleaned.csv` and selects `survived` as the target
- Drops `alive` (a direct leak of the target) and duplicate categorical columns (`class`,
  `embark_town`) that just repeat information already captured elsewhere
- One-hot encodes categorical features (`sex`, `embarked`, `who`, `deck`) with `pd.get_dummies()`
- Splits the data 80/20 into train/test sets with `train_test_split`
- Trains a **Logistic Regression** model and evaluates it with accuracy, precision, recall, and
  F1-score
- Trains a **Decision Tree** for comparison and puts both models' metrics side by side
- Looks at Decision Tree feature importances to see which features drive predictions
- Ends with a written interpretation of the results and ideas for improvement

## How to run it

1. Clone this repo (make sure `titanic_cleaned.csv` from Week 1 is in the same folder, or update
   the file path in the notebook).
2. Install dependencies:
   ```
   pip install pandas scikit-learn jupyter
   ```
3. Launch Jupyter and open the notebook:
   ```
   jupyter notebook week2_classification_model.ipynb
   ```
4. Run all cells top to bottom (`Cell -> Run All`).

## Files

- `week2_classification_model.ipynb` — the main deliverable
- `titanic_cleaned.csv` — the cleaned dataset from Week 1, used as input here
- `README.md` — this file
