import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data_dir = 'datasets'
file_names = os.listdir(data_dir)
print("Pliki w katalogu:", file_names)


classifiers = [
    GaussianNB(),
    KNeighborsClassifier(),
    MLPClassifier(max_iter=1000)
]

n_splits = 2
n_repeats = 5

results = np.zeros((len(file_names), n_splits * n_repeats, len(classifiers)))

for file_idx, file in enumerate(file_names):
    data = np.loadtxt(os.path.join(data_dir, file), delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]

    fold_idx = 0
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits,shuffle = True, random_state=repeat)
        for train_index, test_index in skf.split (X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        for clf_idx, clf in enumerate(classifiers):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[file_idx, fold_idx, clf_idx] = acc

        fold_idx += 1


np.save("wyniki_klasyfikacji.npy", results)
print("Wyniki zapisane do pliku 'wyniki_klasyfikacji.npy'")