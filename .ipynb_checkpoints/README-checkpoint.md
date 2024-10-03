# classification-challenge
Module 13 Challenge
# Spam Detector
## Project overview
This project is a machine learning-based spam detection system that classifies messages as either spam or ham (not spam). The model is built using a combination of Logistic Regression and Random Forest Classifier, and is designed to evaluate and compare their performances on a given dataset.

## Retrieve the Data

The data is located at [https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv](https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv)

Dataset Source: [UCI Machine Learning Library](https://archive.ics.uci.edu/dataset/94/spambase)

Import the data using Pandas. Display the resulting DataFrame to confirm the import was successful.

Predict Model Performance
You will be creating and comparing two models on this data: a Logistic Regression, and a Random Forests Classifier. Before you create, fit, and score the models, make a prediction as to which model you think will perform better. You do not need to be correct!

Write down your prediction in the designated cells in your Jupyter Notebook, and provide justification for your educated guess.
Answer
I believe Logistic Regression will perform better. Given its simplicity and effectiveness with linearly separable data, I expect it to handle the
dataset well, especially if the relationships between the features and target variable are more straightforward. While Random Forest can capture
complex patterns, Logistic Regression’s interpretability and lower risk of overfitting might give it an edge in this case.

## Split the data into Training and Testing sets
## Scale the Features
01-Lession-Plans > 13-Classification > 2 > Activities > 04-Stu_Decision_Tree > Unsolved > malware_tree.ipynb
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)

## Create and Fit a LogisticRegression model
01-Lesson-Plans > 12-Regression > 3 > Activities > 02-Ins_ML_Pipelines > Solved > pipeline_utilities.py
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Create and Fit Random Forest Clasifier Model
Chat GPt review
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)




