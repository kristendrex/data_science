import validation
from sklearn.model_selection import KFold, ShuffleSplit
from numpy import mean

import utils

RANDOM_STATE = 545510477

def get_acc_auc_kfold(X,Y,k=5):
    kf = KFold(n_splits = 5, random_state = RANDOM_STATE,shuffle=True)
    accList = []
    aucList = []
    for train, test in kf.split(X):
        Y_pred = validation.logistic_regression_pred(X[train],Y[train],X[test])
        acc,auc,precision,recall,f1score = validation.classification_metrics(Y_pred, Y[test])
        accList.append(acc)
        aucList.append(auc)
    return round(mean(accList),4), round(mean(aucList),4)

def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
    ss = ShuffleSplit(n_splits=iterNo, test_size=test_percent,random_state = RANDOM_STATE)
    accList = []
    aucList = []
    for train, test in ss.split(X):
        Y_pred = validation.logistic_regression_pred(X[train],Y[train],X[test])
        acc,auc,precision,recall,f1score = validation.classification_metrics(Y_pred, Y[test])
        accList.append(acc)
        aucList.append(auc)
    return round(mean(accList),4), round(mean(aucList),4)


def main():
    X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
    acc_k,auc_k = get_acc_auc_kfold(X,Y)
    acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
    
    with open('kfold_results.txt','a') as writer:
        writer.write("______________________________________________")
        writer.write("\n")
        writer.write("Classifier: Logistic Regression__________")
        writer.write("\n")
        writer.write(("Average Accuracy in KFold CV: "+str(acc_k)))
        writer.write("\n")
        writer.write("Average AUC in KFold CV: "+str(auc_k))
        writer.write("\n")
        writer.write("Average Accuracy in Randomised CV: "+str(acc_r))
        writer.write("\n")
        writer.write(("Average AUC in Randomised CV: "+str(auc_r)))
        writer.write("\n")

if __name__ == "__main__":
    main()