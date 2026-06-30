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