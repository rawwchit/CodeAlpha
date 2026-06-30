import joblib
from src.feature_extractor import extract_features
def predict_emotion(audio_path, model_path, encoder_path):
    # Load trained model
    model = joblib.load(model_path)

    # Load label encoder
    label_encoder = joblib.load(encoder_path)

    # Extract features
    features = extract_features(audio_path)

    # Reshape for prediction
    features = features.reshape(1, -1)

    # Predict
    prediction = model.predict(features)

    # Convert numeric label back to emotion
    emotion = label_encoder.inverse_transform(prediction)
    return emotion[0]