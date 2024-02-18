import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

import utils

def logistic_regression_pred(X_train, Y_train, X_test):
    lr = LogisticRegression()
    lr.fit(X_train,Y_train)
    Y_pred = lr.predict(X_test)
    return Y_pred  

def svm_pred(X_train, Y_train, X_test):
    svm = LinearSVC()
    svm.fit(X_train,Y_train)
    Y_pred = svm.predict(X_test)
    return Y_pred

def decisionTree_pred(X_train, Y_train, X_test):
    dt = DecisionTreeClassifier()
    dt.fit(X_train,Y_train)
    Y_pred = dt.predict(X_test)
    return Y_pred 


#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
    acc = accuracy_score(Y_pred,Y_true)
    auc = roc_auc_score(Y_pred,Y_true)
    precision = average_precision_score(Y_pred,Y_true)
    recall = recall_score(Y_pred,Y_true)
    f1score = f1_score(Y_pred,Y_true)
    return acc,auc,precision,recall,f1score


def display_metrics(classifierName,Y_pred,Y_true):
    acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
    with open('validation_results.txt','a') as writer:
        writer.write("______________________________________________")
        writer.write("\n")
        writer.write("Classifier: "+classifierName)
        writer.write("\n")
        writer.write(("Accuracy: "+str(round(acc,4))))
        writer.write("\n")           
        writer.write("AUC: "+str(round(auc_,4)))
        writer.write("\n")
        writer.write("Precision: "+str(round(precision,4)))
        writer.write("\n")
        writer.write("Recall: "+str(round(recall,4)))
        writer.write("\n")
        writer.write("F1-score: "+str(round(f1score,4)))
        writer.write("\n")
        writer.write("______________________________________________")
        writer.write("\n") 
        
        
def main():
    X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
    X_test, Y_test = utils.get_data_from_svmlight("../data/features_svmlight.validate")

    display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train,X_test),Y_test)
    display_metrics("SVM",svm_pred(X_train,Y_train,X_test),Y_test)
    display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train,X_test),Y_test)

if __name__ == "__main__":
    main()