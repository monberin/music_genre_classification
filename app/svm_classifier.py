import joblib
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report, accuracy_score, confusion_matrix, precision_score, recall_score

import pandas as pd 
import numpy as np
data = pd.read_csv('./features_new.csv')
data = data.iloc[0:, 1:] 
data.head()



y = data['genre']
X = data.loc[:, data.columns != 'genre']

genres = ['blues', 'classical', 'country', 'disco', 'pop', 'hiphop', 'metal', 'reggae','rock']

print(X)
# print(y)

# for ind in range(len(genres)):
#     y = y.replace(genres[ind],ind)

# print(y)

# cols = X.columns
# min_max_scaler = preprocessing.MinMaxScaler()
# np_scaled = min_max_scaler.fit_transform(X)

# X = pd.DataFrame(np_scaled, columns = cols)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



# # svm_grid = SVC(decision_function_shape="ovo")

# param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

# svm_grid = GridSearchCV(SVC(decision_function_shape="ovo"),param_grid,refit=True,verbose=2)

# cv_scores = cross_val_score(svm_grid, X_train, y_train, cv=5)
# #print each cv score (accuracy) and average them
# print(cv_scores)
# print('cv_scores mean:{}'.format(np.mean(cv_scores)))

# svm_grid.fit(X_train, y_train)
# print('fit')

# y_pred = svm_grid.predict(X_test)

# print('Accuracy: ', round(accuracy_score(y_test, y_pred), 5), '\n')

# print(svm_grid.best_params_)

# joblib.dump(svm_grid, './svm_2.joblib')

# loaded_rf = joblib.load("./svm_2.joblib")

# y_pred = loaded_rf.predict(X_test)
# print(X_test)