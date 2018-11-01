# Binary classifier for diagnosing Parkinsons from speech attributes

# Dataset collected by the authors of the following paper:
# Erdogdu Sakar, B., Isenkul, M., Sakar, C.O., Sertbas, A., Gurgen, F., Delil, 
# S., Apaydin, H., Kursun, O., 'Collection and Analysis of a Parkinson Speech 
# Dataset with Multiple Types of Sound Recordings', IEEE Journal of Biomedical 
# and Health Informatics, vol. 17(4), pp. 828-834, 2013.

# Dataset provided by the UCI machine learning repository:
# Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository 
# [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
# School of Information and Computer Science.

### 1. Imports
# data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# machine learning support
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

# machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

### 2. Load and describe data
dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/\
parkinsons/parkinsons.data"
data_raw = pd.read_csv(dataset_url)
print("Available features:\n", data_raw.columns.values)
data_raw.info()
# 24 columns/features: 22 floats, 1 object/string ('name'), 1 int ('status'=y)
# 195 rows/examples
# no missing or null data to imputate
data_raw['status'].value_counts()
# our label 'status' (parkinsons diagnosis) has 147 positive examples and 48 negative,
# since this distribution does not terribly favor one class over the other
# do not need to weigh false positives and false negatives differently,
# use accuracy to score

### 3. Split data ...
# ... into labels (y) and features (X) ...
y = data_raw.status
X = data_raw.drop(['status', 'name'], axis=1) # subject's name irrelevant to diagnosis
# ...and into training (80%) and testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)

### 4. Preprocess
# save mean and std for training set
scaler = preprocessing.StandardScaler().fit(X_train)
# use those values to standardize training set
X_train_scaled = scaler.transform(X_train)
print("Training Mean:", X_train_scaled.mean(axis=0).round(2)) # confirm feature means are 0
print("Training Std:", X_train_scaled.std(axis=0).round(2)) # confirm feature stds are 1
# use the same values to standardize testing set
X_test_scaled = scaler.transform(X_test)
print("Testing Mean:", X_test_scaled.mean(axis=0).round(2)) # confirm feature means are near 0
print("Testing Std:", X_test_scaled.std(axis=0).round(2)) # confirm feature stds are near 1
print('_' * 40)

### 5. Algorithm I: Logistic Regression
# cross-validation for choosing hyperparameters, then fitting
# hyperparameters: C (inverse regularization parameter)
logreg_hyperparameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
logreg_clf = GridSearchCV(LogisticRegression(), logreg_hyperparameters, 
                          cv=10, scoring='accuracy')
logreg_clf.fit(X_train_scaled, y_train)
print("logreg:\nbest hyperparameters are %s with best cv accuracy of %0.3f"
      % (logreg_clf.best_params_, logreg_clf.best_score_)) # C=100 acc=0.859
# evaluate model on training and testing sets
logreg_train_pred = logreg_clf.predict(X_train_scaled)
print("training accuracy: %0.3f" % (accuracy_score(y_train, logreg_train_pred))) # =0.923
logreg_test_pred = logreg_clf.predict(X_test_scaled)
logreg_test_acc = (accuracy_score(y_test, logreg_test_pred))
print("testing accuracy: %0.3f" % logreg_test_acc) # =0.821
print('_' * 40)

### 6. Algorithm II: Support Vector Machine (Gaussian/RBF kernel)
# cv and fitting
# hyperparameters: C (inverse regularization, specifically allowance for soft margins)
#                  gamma (inverse regularization, kernel coeff. for similarity droppoff)
svc_hyperparameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                       'gamma': [0.001, 0.01, 0.1, 1]}
svc_clf = GridSearchCV(SVC(kernel='rbf'), svc_hyperparameters, cv=10, 
                       scoring='accuracy')
svc_clf.fit(X_train_scaled, y_train)
print("svc:\nbest hyperparameters are %s with best cv accuracy of %0.3f"
      % (svc_clf.best_params_, svc_clf.best_score_)) # C=100 gamma=0.1 acc=0.923
# evaluate
svc_train_pred = svc_clf.predict(X_train_scaled)
print("training accuracy: %0.3f" % (accuracy_score(y_train, svc_train_pred))) # =1.00
svc_test_pred = svc_clf.predict(X_test_scaled)
svc_test_acc = (accuracy_score(y_test, svc_test_pred))
print("testing accuracy: %0.3f" % svc_test_acc) # =0.872
print('_' * 40)

### 7. Algorithm III: Random Forest
# cv and fitting
# hyperparameters: max_features (max no. of features to consider for a split)
#                  max_depth (max no. of layers to include)
#                  (considerably more hyperparamaters exist but will start with these)
rf_hyperparameters = {'max_features': [3, 5, 9], # function of n_features
                      'max_depth': [None, 12, 6]}
rf_clf = GridSearchCV(RandomForestClassifier(n_estimators=200, random_state=23),  
                      rf_hyperparameters, cv=10, scoring='accuracy')
rf_clf.fit(X_train_scaled, y_train)
print("rf:\nbest hyperparameters are %s with best cv accuracy of %0.3f"
      % (rf_clf.best_params_, rf_clf.best_score_)) 
# max_depth=6 max_features=3 acc=0.910
# evaluate
rf_train_pred = rf_clf.predict(X_train_scaled)
print("training accuracy: %0.3f" % (accuracy_score(y_train, rf_train_pred))) # =1.00
rf_test_pred = rf_clf.predict(X_test_scaled)
rf_test_acc = (accuracy_score(y_test, rf_test_pred))
print("testing accuracy: %0.3f" % rf_test_acc) # =0.923
print('_' * 40)

### 8. Algorithm IV: K-Nearest Neighbors
# cv and fitting
# hyperparameters: n_neighbors (k, no. of nearest neighbors to consider)
#                  weights (weights to the considered neighbors in prediction, 
#                           uniform or by distance)
knn_hyperparameters = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13], 
                       'weights': ['uniform', 'distance']}
knn_clf = GridSearchCV(KNeighborsClassifier(), knn_hyperparameters, 
                        cv=10, scoring='accuracy')
knn_clf.fit(X_train_scaled, y_train)
print("knn:\nbest hyperparameters are %s with best cv accuracy of %0.3f"
      % (knn_clf.best_params_, knn_clf.best_score_)) 
# n_neighbors=1 weights='uniform acc=0.949
# evaluate
knn_train_pred = knn_clf.predict(X_train_scaled)
print("training accuracy: %0.3f" % (accuracy_score(y_train, knn_train_pred))) # =1.00
knn_test_pred = knn_clf.predict(X_test_scaled)
knn_test_acc = (accuracy_score(y_test, knn_test_pred))
print("testing accuracy: %0.3f" % knn_test_acc) # =0.872
print('_' * 40)

### 9. Algorithm V: Gaussian Naive-Bayes
# not much potential for hyperparameter tuning
gnb_clf = GaussianNB()
gnb_clf.fit(X_train_scaled, y_train)
gnb_train_pred = gnb_clf.predict(X_train_scaled)
print("gnb:\ntraining accuracy: %0.3f" 
      % (accuracy_score(y_train, gnb_train_pred))) # =0.699
gnb_test_pred = gnb_clf.predict(X_test_scaled)
gnb_test_acc = (accuracy_score(y_test, gnb_test_pred))
print("testing accuracy: %0.3f" % gnb_test_acc) # =0.718
print('_' * 40)

### 10. Compare classifiers
# compare testing accuracies of the above classifiers
compare_models = pd.DataFrame({
        'Model': ['Logistic Regression', 'SVM', 'Random Forest', 'K-NN', 'G-NB'],
        'Testing Accuracy': [logreg_test_acc, svc_test_acc, rf_test_acc, 
                             knn_test_acc, gnb_test_acc]})
compare_models.sort_values(by='Testing Accuracy', ascending=False)
print('_' * 40)
# random forest has the best testing score and maybe also the most potential for 
# further hyperparameter tuning (several influential untuned hyperparameters 
# in current predictor), will revisit random forest algorithm

### 11. Random Forest - revisited
# new set of hyperparameters to tune
# hyperparameters: max_features (max no. of features to consider for a split)
#                  max_depth (max no. of layers to include)
#                  min_samples_split (min no. of samples required to make a split)
#                  min_samples_leaf (min no. of samples required at each leaf node)
#                  n_estimators (no. of trees in the forest)
#                  bootstrap (sample points with or without replacement)
rev_parameters = {'max_features': [3],     # value from earlier classifier
                   'max_depth': [6],        # value from earlier classifier
                   'min_samples_split': [2, 4, 6],
                   'min_samples_leaf': [1, 3, 5],
                   'n_estimators': [160, 180, 200, 220],
                   'bootstrap': [True, False]}
rf_rev_clf = GridSearchCV(RandomForestClassifier(random_state=23),
                           rev_parameters, cv=10, scoring='accuracy')
rf_rev_clf.fit(X_train_scaled, y_train)
print("rf_rev:\nbest hyperparameters are\n%s\nwith best cv accuracy of %0.3f"
      % (rf_rev_clf.best_params_, rf_rev_clf.best_score_))
# min_samples_split=2 (Default), min_samples_leaf=1 (default), bootstrap=True (default)
# n_estimators=160, CVacc=0.910
# evaluate
rf_rev_train_pred = rf_rev_clf.predict(X_train_scaled)
print("training accuracy: %0.3f" % (accuracy_score(y_train, rf_rev_train_pred))) # =1.000
rf_rev_test_pred = rf_rev_clf.predict(X_test_scaled)
rf_rev_test_acc = (accuracy_score(y_test, rf_rev_test_pred))
print("testing accuracy: %0.3f" % rf_rev_test_acc) # =0.923 
print('_' * 40)  
# Tuning these additional hyperparameters did not see any gain in score and three
# of the hyperparameters remained at the default values used in the last rf classifier.
# With the testing set size, there are likely just a few samples that both classifiers
# are failing to predict. Future project could filter outliers more, maybe resolving
# those few points, or add polynomial features to explain them, but for the scope
# of the current project a classifier with 92.3% accuracy is acceptable. Will 
# continue working with the random forest classifier from section 7.

### 12. Comparing features
# compare feature contributions to prediction of each class for the original 
# random forest classifier
feature_importances = rf_clf.best_estimator_.feature_importances_
feature_comp = pd.DataFrame(feature_importances, index=X_train.columns,
                                  columns=['Importance'])
sorted_feature_comp = feature_comp.sort_values('Importance', ascending=True)
rev_feature_comp = feature_comp.sort_values('Importance', ascending=False)
print("Features Ranked by Importance to Random Forest Classifier:\n\n", 
      rev_feature_comp)
comp_plot = plt.barh(np.arange(22), sorted_feature_comp.Importance, 
                   tick_label=sorted_feature_comp.index)
# The most important features for prediction are then PPE, spread1, MDVP:Fo(Hz),
# spread2, and MDVP:Fhi(Hz). The least important are Shimmer:APQ3, Shimmer:DDA,
# MDVP:Shimmer, MDVP:Jitter(%), and MDVP:Shimmer(dB).

### 13. Export classifier
joblib.dump(rf_clf, 'parkinsons_speech_rf.pkl')











