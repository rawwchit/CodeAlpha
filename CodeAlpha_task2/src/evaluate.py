from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
def evaluate_model(model, X_test, y_test, label_encoder):
    # Predictions
    y_pred = model.predict(X_test)
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Accuracy")
    print(f"Accuracy : {accuracy:.4f}")
    # Classification Report
    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_
        )
    )
    # Confusion Matrix
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))