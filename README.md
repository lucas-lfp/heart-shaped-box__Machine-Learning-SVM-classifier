
This project aims at building an app that provides a probability of cardiovascular disease based on user's input. Model was trained using cardiovascular disease dataset from [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).
# Heart Shaped Box - SVM Classifier for Cardiovascular Diseases Detection

![Banner Image](link/to/banner/image)

## Overview
This project focuses on the development of a Support Vector Machine (SVM) classifier for the detection of cardiovascular diseases. The SVM model achieved an accuracy of 73.6% and is deployed on a Streamlit app, allowing users to test it with their own data.
Model was deployed on a Streamlit app, with two main functionnalitites:
- User can browse through a collection of dataviz to gain relevant insight on adult population characteristics, regarding cardiovascular diseases.
- User can test the model with their own data to gain insight on their cardiovascular status.

--> **Visit the [streamlit app](https://lucas-lfp-hsb.streamlit.app/).**

## Objectives
- To understand the relationships between clinical features and, particularly, their connection to the target feature through relevant visualisations.
- To find a suitable machine learning model for this application
- To evaluate the impact of hyperparameters on the model's performances
- To put the model into production so it can treat user's input

## Skills Used in This Project

- Project was developped in **Python**, using **HTML/CSS** for formatting. 
- Libraries used in this project: **Pandas**, **NumPy**, **SciPy**, **Matplotlib**, **Seaborn**, **Scikit-learn**, **joblib**, **streamlit**.

## Method

Development of this application was made following a four-steps plan, as described below.

1. **Data Analysis** 

2. **Model Selection** - Several models were evaluated on this dataset: Logistic Regression, Random Forest, Support Vector Machines, K-Nearest Neighbors and AdaBoost. Support Vector Machines was selected for this project, as it presented the best accuracy (73.2%), a good rate of correct predictions for Class 0 (77.6%) and an acceptable rate for Class 1 (68.8%).

3. **Hyperparameter Tuning** - SVM model was tuned to improve its performances, several combinations of kernel, C and gamma were tested. RBF kernel, with C = 0.1 and gamma = 10 led to good and balanced performances, and was the best combination for a high recall without a significant loss in accuracy.

4. **Deployment**

## Results
- SVM Classifier Accuracy: 73.6%.
- Usefull set of visualisation produced.
