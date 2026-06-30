import joblib
from src.data_loader import load_dataset
from src.feature_extractor import extract_features
from src.preprocessing import preprocess
from src.train import train_model
from src.evaluate import evaluate_model

import numpy as np


def main():

    # Load dataset
    df = load_dataset("data/raw/RAVDESS")

    print(f"Total audio files: {len(df)}")

    # Extract features
    X = []
    y = []

    for index, row in df.iterrows():

        features = extract_features(row["filepath"])

        X.append(features)
        y.append(row["emotion"])

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(df)} files")

    X = np.array(X)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test, label_encoder = preprocess(X, y)
    print("\nTrain-Test Split")
    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")
    print("\nTraining Random Forest Model...")
    
    model = train_model(X_train, y_train)
    print("Training Completed!")
    
    evaluate_model(
    model,
    X_test,
    y_test,
    label_encoder
    )
    print("\nSaving Model...")
    joblib.dump(model, "models/emotion_model.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")
    print("Model Saved Successfully!")

    print("\nDataset Created Successfully")
    print(f"Feature Matrix Shape : {X.shape}")
    print(f"Labels Shape         : {y.shape}")

    print("\nEmotion Classes:")
    print(np.unique(y))


if __name__ == "__main__":
    main()
    print("\nPipeline Completed Successfully!")