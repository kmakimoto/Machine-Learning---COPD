# Code for Machine Learning  Models with CT imagining features for Classifying Binary COPD Status
# By: Kalysta Makimoto, Bsc
# Email: kalysta.makimoto@torountomu.ca
# Last Modified: May 30th, 2023

# Import Required Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from math import sqrt
import shap

# BEFORE RUNNING:
    # Update file location for the features (Line 42)
    # Upadate file location for the true labels (Line 49)
    # Update export files names (Line 228 and Line 229)

# Load Files: Features and Labels
# All Participants (1204)
    # Demo + qCT + Radiomics Features
Features = pd.DataFrame(pd.read_excel(r'/Users/XXX.xlsx'))  # Load Features to investigate as an excel file
feature_names = np.array(Features.columns)  # Obtain List of Features included

# Import True Labels [Labels: 0 = no COPD; 1 = COPD]
    # Labels for the whole cohort included
AllLabel = pd.DataFrame(pd.read_excel(r'/Users/XXX.xlsx'))  # Load labels for all the participants included - keep in same index as the feature sets

# Calculate weights for the models: the ratio of 0 class labels to 1 (binary) [Labels: 0 = no COPD; 1 = COPD]
odds = (len(np.array(AllLabel)) - sum(np.array(AllLabel))) / sum(np.array(AllLabel))
weights = {0:odds[0], 1:1}

# Split the dataset into a training (75%) and testing (25%)
Training_features, Test_features, Training_Label, Test_Label = train_test_split(Features, AllLabel,
                                                                                stratify=AllLabel,
                                                                                test_size=0.25,
                                                                                random_state=1)
Training_features = pd.DataFrame(Training_features)    # Convert the features into a DataFrame
Training_features.columns = feature_names   # Obtain list of training features names
Test_features = pd.DataFrame(Test_features)     # Convert the features into a DataFrame
Test_features.columns = feature_names   # Obtain list of testing features names

# Remove Outliers from Training Cohort
for features in feature_names:  # investigate each feature independently
    upper_limit = Training_features[features].mean() + 2*Training_features[features].std()  # Determine the upper 2 SD range
    lower_limit = Training_features[features].mean() - 2*Training_features[features].std()  # Determine the lower 2 SD range
    OutlierRemoved_Training_features = Training_features[(Training_features[features] < upper_limit) & (Training_features[features] > lower_limit)]     # Determines which features are outliers and puts them in a list

# Separate Training Labels and Features after Outliers are removed
Training_Label = Training_Label.T   # Transpose the Training labels
Cleaned_Training_Label_indices = np.array(OutlierRemoved_Training_features.index)   # Create of array of the indices for the participants to remove
Cleaned_Training_Label = np.array(Training_Label[Cleaned_Training_Label_indices])   # Remove participants identified as an outlier
Cleaned_Training_Label = Cleaned_Training_Label.T    # Transpose the Training labels

# Scale / Normalize Data
Cleaned_Training_features = StandardScaler().fit_transform(OutlierRemoved_Training_features)    # Scale the features
Cleaned_Training_features = pd.DataFrame(Cleaned_Training_features)
Cleaned_Training_features.columns = feature_names

Test_features = StandardScaler().fit_transform(Test_features)   # Scale the features
Test_features = pd.DataFrame(Test_features)
Test_features.columns = feature_names

# Determine highly correlated features (test with above 0.9)
corr_matrix = Cleaned_Training_features.corr()  # Create feature correlation matrix
correlated_features = set()
for i in range(len(corr_matrix.columns)):
   for j in range(i):
        if abs(corr_matrix.iloc[i,j])>0.9:      # Identify which features have a correlation coefficient about 0.90
                colname = corr_matrix.columns[i]    # Creates a list of the highly correlated features
                correlated_features.add(colname)

# Remove highly correlated features (keeping only 1 from list)
if len(correlated_features) > 0:    # If there are highly correlated features, run the code
    keep = list(correlated_features)[0]     # Keep 1 feature to prevent the loss of data
    correlated_features.remove(keep)    # Identify which features to remove
    Test_features.drop(labels=correlated_features, axis=1, inplace=True)    # Remove correlated features from the test set
    Cleaned_Training_features.drop(labels=correlated_features, axis=1, inplace=True)    # Remove correlated features from the train set
    feature_names = np.array(Cleaned_Training_features.columns)     # Create new list of features included

# Implement feature selection methods
All_FS_Scores = []      # Create  matrix for feature selection values
All_FS_Scores = pd.DataFrame(All_FS_Scores)     # Convert to a DataFrame

# Implement Random Forest for feature selected with the training dataset
Train_FS_rf = RandomForestClassifier(n_estimators=100, random_state=0, class_weight=weights)
Train_FS_rf.fit(Cleaned_Training_features, Cleaned_Training_Label)
Train_FS_rf_value = pd.DataFrame(Train_FS_rf.feature_importances_)      # Obtain list of feature importance values/scores
All_FS_Scores['Train_FS_rf_value'] = Train_FS_rf_value      # Add features values to matrix with all values for each method

# Implement ExtraTree for feature selected with the training dataset
Train_FS_extratree = ExtraTreesClassifier(n_estimators=100, random_state=0, class_weight=weights)
Train_FS_extratree.fit(Cleaned_Training_features, Cleaned_Training_Label)
Train_FS_extratree_value = pd.DataFrame(Train_FS_extratree.feature_importances_)    # Obtain list of feature importance values/scores
All_FS_Scores['Train_FS_extratree_value'] = Train_FS_extratree_value        # Add features values to matrix with all values for each method

# Implement Gradient Boosting Decision Tree for feature selected with the training dataset
Train_FS_gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
Train_FS_gbdt.fit(Cleaned_Training_features, Cleaned_Training_Label)
Train_FS_gbdt_value = pd.DataFrame(Train_FS_gbdt.feature_importances_)  # Obtain list of feature importance values/scores
All_FS_Scores['Train_FS_gbdt_value'] = Train_FS_gbdt_value      # Add features values to matrix with all values for each method

# Implement L-based Logistic Regression for feature selected with the training dataset
Train_FS_LR = LogisticRegression(penalty='l2', max_iter=1000, random_state=0, class_weight=weights)
Train_FS_LR.fit(Cleaned_Training_features, Cleaned_Training_Label)
Train_FS_LR_value = pd.DataFrame(Train_FS_LR.coef_)     # Obtain list of feature importance values/scores
All_FS_Scores['Train_FS_LR_value'] = Train_FS_LR_value.T        # Add features values to matrix with all values for each method

# Implement LASSO for feature selected with the training dataset
Train_FS_lasso = Lasso(alpha=0.01, fit_intercept=False, normalize=False, random_state=0)
Train_FS_lasso.fit(Cleaned_Training_features, Cleaned_Training_Label)
Train_FS_lasso_value = pd.DataFrame(Train_FS_lasso.coef_)       # Obtain list of feature importance values/scores
All_FS_Scores['Train_FS_lasso_value'] = Train_FS_lasso_value        # Add features values to matrix with all values for each method

# Implement Elastic Net for feature selected with the training dataset
Train_FS_en = ElasticNet(alpha=0.01, fit_intercept=False, normalize=False, random_state=0)
Train_FS_en.fit(Cleaned_Training_features, Cleaned_Training_Label)
Train_FS_en_value = pd.DataFrame(Train_FS_en.coef_)     # Obtain list of feature importance values/scores
All_FS_Scores['Train_FS_en_value'] = Train_FS_en_value      # Add features values to matrix with all values for each method

# Implement classification methods
FS_names = np.array(All_FS_Scores.columns)      # Obtain a list of the feature selection methods
n_features = [5]    # Identify number of features to selected from the feature selection methods

# Create an empty table for the models investigate and the performance metrics
Classifier_result_table = pd.DataFrame(columns=['classifiers', 'fs', 'best_param', 'best_auc_CV','No_fts', 'auc', 'lower', 'upper','f1', 'acc', 'precision', 'recall', 'features'])
# Create an empty table for the probabilities for the test set labels - used for DeLong's Test
Probabilities_table = pd.DataFrame(columns=['classifier', 'fs', 'proba'])

# Define classifiers to be investigated
classifiers = [LogisticRegression(random_state=0, class_weight=weights),
               KNeighborsClassifier(),
               RandomForestClassifier(class_weight=weights, random_state=0),
               MLPClassifier(random_state=0),
               svm.SVC(class_weight=weights, random_state=0, probability=True)]

# Define GridSearchCV parameters to test, match index with classifiers index
parameters = [{'solver': ['lbfgs', 'liblinear','newton-cg'], 'C': [0.1, 1, 10]},
              {'n_neighbors': [3, 5, 8], 'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto']},
              {'n_estimators': [10, 50, 100, 150], 'max_features': ['sqrt', 'log2'], 'max_depth': [5, 10, 15]},
              {'hidden_layer_sizes': [10, 50, 100], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam']},
              {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]

# Create dataframe to contain the performance metrics, model combinations and features selected
for names in FS_names:  # Investigate each feature selection method individually
    Scores = [feature_names, All_FS_Scores[names].T]
    Scores = pd.DataFrame(Scores)
    Scores = Scores.T
    Scores.columns = ['Feature', 'Score']
    Scores = Scores.reindex(Scores.Score.abs().sort_values(ascending=False).index)  # Sort feature importance values by magnitude to determine most important features

    for n in n_features:    # Investigate different number of input features

        TopFeatures = Scores[0:n]       # Selected to top n features to investigate
        Features_list = TopFeatures.Feature

        Training_Features = Cleaned_Training_features[Features_list]    # Selects the top n features in the training dataset
        Test_Features = Test_features[Features_list]        # Selects the top n features in the testing dataset

        for x in range(len(classifiers)):       # Investigate each classifier individually

            model = classifiers[x]  # Identify the classifier to be investigated
            print(model)
            param = parameters[x]   # Identify the GridSearchCV parameters to investigate
            grid_search = GridSearchCV(model, param, cv=5, scoring='roc_auc')   # Run and fit the grid search to identify top parameters
            grid_search.fit(Training_Features, Cleaned_Training_Label)      # Trains the models with the training dataset

            clf = grid_search.best_estimator_       # State the top parameters for the models
            print(clf)

            yproba = clf.predict_proba(Test_Features)   # Calculates the predicted probability of each class with the testing dataset
            ypred = clf.predict(Test_Features)      # Calculates the predicted label for testing dataset

            auc = roc_auc_score(Test_Label, yproba[:, 1], average='weighted')   # Calculates the AUC values for the testing dataset
            f1 = f1_score(Test_Label, ypred, average='weighted')    # Calculates the F1-score values for the testing dataset
            acc = accuracy_score(Test_Label, ypred)     # Calculates the accuracy values for the testing dataset
            precision = precision_score(Test_Label, ypred, average='weighted')      # Calculates the precision values for the testing dataset
            recall = recall_score(Test_Label, ypred, average='weighted')        # Calculates the recall values for the testing dataset

            # Calculates the upper and lower 95% CI for the AUC values
            N1 = Test_Label.sum()
            N2 = len(Test_Label) - N1
            Q1 = auc / (2 - auc)
            Q2 = 2 * auc ** 2 / (1 + auc)
            SE_auc = sqrt((auc * (1 - auc) + (N1 - 1) * (Q1 - auc ** 2) + (N2 - 1) * (Q2 - auc ** 2)) / (N1 * N2))
            lower = auc - 1.96 * SE_auc
            upper = auc + 1.96 * SE_auc

            # Adds the model parameters and performance metrics to the table
            Classifier_result_table = Classifier_result_table.append({'classifiers': model,
                                                                      'fs': names,
                                                                      'best_param': grid_search.best_params_,
                                                                      'best_auc_CV': grid_search.best_score_,
                                                                      'No_fts': n,
                                                                      'auc': auc,
                                                                      'lower': lower,
                                                                      'upper': upper,
                                                                       'f1': f1,
                                                                       'acc': acc,
                                                                       'precision': precision,
                                                                       'recall': recall,
                                                                      'features': TopFeatures.Feature},
                                                                       ignore_index=True)

            # Adds the testing dataset probabilities to the table
            Probabilities_table = Probabilities_table.append({'classifier': model,
                                                              'fs': names,
                                                              'proba': yproba[:, 1]},
                                                             ignore_index=True)

# Export the Results Table and Probabilities Table results
Classifier_result_table.to_excel(r'/Users/XXX.xlsx')
Probabilities_table.to_excel(r'/Users/XX.xlsx')