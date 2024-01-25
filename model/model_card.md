# Model Card

## Model Details
Author: Jeroen van Lier
Model: Random Forest
Parameters: Used the default hyperparameters in scikit-learn 1.3.2

## Intended Use
The model is intended to be used to predict whether a person makes over 50K a year.

## Data
The Data was downloaded from the UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/20/census+income

A model pipeline was created to process the data which includes the following steps:
Numerical data was scaled using the StandardScaler 
Catagorical data was one-hot encoded using the OneHotEncoder 
Label data was encoded using the LabelEncoder

## Metrics
Precision: 0.7211
Recall: 0.6195
FBeta: 0.6664
Accuracy: 0.8534
F1: 0.6664

The metrics for each of slice of the catagorical columns are shown in model/slice_output.txt

## Ethical Considerations
This model uses income data, which may reflect societal biases. 
