from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
def preprocess(X, y):
    # Convert emotion names into numbers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test, label_encoder