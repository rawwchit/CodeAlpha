"""
Data Loader Module

This module is responsible for:
1. Finding all audio files
2. Extracting emotion labels from filenames
3. Preparing the dataset for feature extraction
"""

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
    """
    Load the RAVDESS dataset and extract
    file paths along with their emotion labels.

    Parameters
    ----------
    dataset_path : str
        Path to the RAVDESS dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame containing file paths and emotion labels.
    """
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