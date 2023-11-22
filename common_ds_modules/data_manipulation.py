from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

def get_numerical_categorical_variables(train, test, id_column):
    categorical_variables = []
    numerical_variables = []

    for column in train.columns:
        if train[column].dtype == 'object':
            categorical_variables.append(column)
        else:
            if id_column != column:
                if len(train[column].unique()) <= 10:
                    categorical_variables.append(column)
                else: # if you don't have the check, id columns have a lot of possible values
                    numerical_variables.append(column)

    return numerical_variables, categorical_variables

def get_train_test_model_data(train, test, numerical_variables, categorical_variables, target, target2=None):
    if target2 is not None:
        y_train = train[target2]
    else:
        y_train = train[target2]

    train_model_data = train.copy()

    numerical_data = train_model_data[[var for var in numerical_variables if target not in var]] # might need to fix this later
    categorical_data = train_model_data[categorical_variables]
    categorical_data = pd.get_dummies(categorical_data)
    train_model_data = pd.concat([numerical_data, categorical_data], axis=1)

    test_model_data = test.iloc[:, :].copy()
    numerical_data = test_model_data[[var for var in numerical_variables if target not in var]]
    categorical_data = test_model_data[categorical_variables]
    categorical_data = pd.get_dummies(categorical_data)
    test_model_data = pd.concat([numerical_data, categorical_data], axis=1)

    return train_model_data, test_model_data


def get_variables(df, ignore_variables, id_column, limit=15):
    categorical_variables = []
    discrete_numerical_variables = []
    continuous_numerical_variables = []

    for column in df.columns:
        if column not in ignore_variables:
            if df[column].dtype == 'O':
                categorical_variables.append(column)
            else:
                if column != id_column:
                    if len(df[column].unique()) <= limit:  # not a hard and fast rule, can be changed
                        discrete_numerical_variables.append(column)
                    else:
                        continuous_numerical_variables.append(column)

    return categorical_variables, discrete_numerical_variables, continuous_numerical_variables

