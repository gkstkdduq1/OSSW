# PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(dataset_path):
    return pd.read_csv(dataset_path)


def dataset_stat(dataset_df):
    return sum(dataset_df.columns != 'target'), sum(dataset_df.target == 0), sum(dataset_df.target == 1)


def split_dataset(dataset_df, testset_size):
    x = dataset_df.iloc[:, :-1]
    y = dataset_df.iloc[:, -1]
    return train_test_split(x, y)


def decision_tree_train_test(x_train, x_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return get_metrics(y_pred, y_test)


def random_forest_train_test(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return get_metrics(y_pred, y_test)


def svm_train_test(x_train, x_test, y_train, y_test):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return get_metrics(y_pred, y_test)


def get_metrics(y_pred, y_true):
    return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)


def print_performances(acc, prec, recall):
    # Do not modify this function!
    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", recall)


if __name__ == '__main__':
    # Do not modify the main script!
    sys.argv = [0, '.\heart.csv', 0.2]
    data_path = sys.argv[1]
    data_df = load_dataset(data_path)

    n_feats, n_class0, n_class1 = dataset_stat(data_df)
    print("Number of features: ", n_feats)
    print("Number of class 0 data entries: ", n_class0)
    print("Number of class 1 data entries: ", n_class1)

    print("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
    x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

    acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
    print("\nDecision Tree Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
    print("\nRandom Forest Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
    print("\nSVM Performances")
    print_performances(acc, prec, recall)
