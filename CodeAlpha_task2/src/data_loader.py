from pathlib import Path
import pandas as pd

EMOTION_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}

def load_dataset(dataset_path: str) -> pd.DataFrame:
    records = []

    dataset_path = Path(dataset_path)

    audio_files = dataset_path.rglob("*.wav")
        
    for audio_file in audio_files:
        filename = audio_file.name
        parts = filename.split("-")
        emotion_code = parts[2]
        emotion = EMOTION_MAP[emotion_code]
        records.append({
        "filepath": str(audio_file),
        "emotion": emotion
    })
    df = pd.DataFrame(records)
    return df