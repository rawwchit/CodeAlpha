from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def evaluate_model(model,x_test,y_test):
    pred = model.predict(x_test)
    print()
    print("Accurcay:",accuracy_score(y_test, pred))
    print("Precision:",precision_score(y_test, pred))
    print("Recall:",recall_score(y_test, pred))
    print("F1 Score:",f1_score(y_test, pred))
    print("ROC AUC:",roc_auc_score(y_test, pred))
    print()
    print("Confusion Matrix")
    print(confusion_matrix(y_test, pred))
    