# Model Card

This model was developed for the Udacity Nanodegree on Machine Learning DevOps Engineer. 

## Model Details
Author: Jeroen van Lier
Model: Random Forest
Parameters: Used the default hyperparameters in scikit-learn 1.3.2

## Intended Use
The model is intended to be used to predict whether a person makes over 50K a year.

## Training Data
The Data was downloaded from the UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/20/census+income

A model pipeline was created to process the data which includes the following steps:
* Numerical data was scaled using the StandardScaler 
* Catagorical data was one-hot encoded using the OneHotEncoder 
* Label data was encoded using the LabelEncoder

## Evaluation Data
The data was split into a training set and a test set using a 80/20 split for model evaluation.

## Metrics
The model was evaluated using the following metrics:
Precision: 0.7211
Recall: 0.6195
FBeta: 0.6664
Accuracy: 0.8534
F1: 0.6664

The metrics for each of slice of the catagorical columns are shown in model/slice_output.txt

## Ethical Considerations
This model uses income data, which may reflect societal biases. Please see the UCI Machine Learning Repository for more information about the data to understand the potential bias of the data.

# Caveats and Recommendations
The model's predictions are based on historical data, which may not accurately represent current or future trends. Additionally, the model's performance might vary across different demographic groups. Please refer to the model/slice_output.txt file for more information about the model's performance on different demographic groups.