from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_logistic(X_train,y_train):
    model = LogisticRegression(
                    max_iter=2000
            )
    model.fit(
            X_train,
            y_train
    )
    return model

def train_dt(X_train,y_train):
    model = DecisionTreeClassifier(
            max_depth=5,
            random_state=42
    )
    model.fit(
            X_train,
            y_train
    )
    return model

def train_rf(X_train,y_train):
    model = RandomForestClassifier(
                random_state=42
    )
    model.fit(
            X_train,
            y_train
    )
    return model