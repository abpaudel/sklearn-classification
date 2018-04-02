import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn import svm, grid_search
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

data = np.genfromtxt('input3.csv', delimiter = ',', skip_header = 1)
x = data[:, :2]
y = data[:, 2]
color = ['blue' if l == 1 else 'red' for l in y]
plt.scatter(x[:, 0], x[:, 1], color = color)

X_train_cv, X_test, y_train_cv, y_test = train_test_split(x, y, test_size = 0.4, stratify = y)

params_svm_linear = {'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'kernel': ['linear']}
params_svm_polynomial = {'C' : [0.1, 1, 3], 'degree' : [4, 5, 6], 'gamma' : [0.1, 0.5], 'kernel': ['poly']}
params_svm_rbf = {'C' : [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma' : [0.1, 0.5, 1, 3, 6, 10],'kernel': ['rbf']}
params_logistic = {'C':  [0.1, 0.5, 1, 5, 10, 50, 100]}
params_knn = {'n_neighbors' : list(range(1,51)), 'leaf_size' : list(range(5,61,5))}
params_decision_tree = {'max_depth' : list(range(1,51)), 'min_samples_split' : list(range(2,11))}
params_random_forest = {'max_depth' : list(range(1,51)), 'min_samples_split' : list(range(2,11))}

parameters = [('svm_linear', params_svm_linear, svm.SVC()),
              ('svm_polynomial', params_svm_polynomial, svm.SVC()),
              ('svm_rbf', params_svm_rbf, svm.SVC()),
              ('logistic', params_logistic, LogisticRegression()),
              ('knn', params_knn, KNeighborsClassifier()),
              ('decision_tree', params_decision_tree, tree.DecisionTreeClassifier()),
              ('random_forest', params_random_forest, RandomForestClassifier())]

file = open('output3.csv', 'w')

for model_name, param_grid, est in parameters:
    model = grid_search.GridSearchCV(estimator = est, param_grid = param_grid, cv = 5)
    model.fit(X_train_cv, y_train_cv)
    best_score=model.best_score_
    test_score=model.score(X_test, y_test)
    
    file.write(model_name + ',' + str(best_score) + ',' + str(test_score) + '\n')
    
    h = 0.02
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    Z = model.predict(np.c_[x1.ravel(), x2.ravel()])
    Z = Z.reshape(x1.shape)
    positive_ex = data[np.where(data[:, 2] == 1)]
    negative_ex = data[np.where(data[:, 2] == 0)]
    plt.plot(positive_ex[:, 0], positive_ex[:, 1], 'ro', negative_ex[:, 0], negative_ex[:,1], 'bo')
    plt.contourf(x1, x2, Z, cmap = plt.cm.coolwarm, alpha = 0.8)
    plt.show()
    
file.close()