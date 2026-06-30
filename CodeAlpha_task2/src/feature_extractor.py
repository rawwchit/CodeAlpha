"""
Feature Extraction Module
This module extracts audio features from speech files.
"""
import librosa
import numpy as np
def extract_features(audio_path: str) -> np.ndarray:
    """
Feature Extraction Module

This module extracts audio features from speech files.
"""

import librosa
import numpy as np

def extract_features(audio_path: str) -> np.ndarray:
    # Load audio file
    signal, sample_rate = librosa.load(audio_path, sr=None)

    # MFCC (40 coefficients)
    mfcc = np.mean(
        librosa.feature.mfcc(
            y=signal,
            sr=sample_rate,
            n_mfcc=40
        ).T,
        axis=0
    )

    # Chroma Features
    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=signal,
            sr=sample_rate
        ).T,
        axis=0
    )

    # Mel Spectrogram
    mel = np.mean(
        librosa.feature.melspectrogram(
            y=signal,
            sr=sample_rate
        ).T,
        axis=0
    )

    # Zero Crossing Rate
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(signal).T,
        axis=0
    )

    # Root Mean Square Energy
    rms = np.mean(
        librosa.feature.rms(y=signal).T,
        axis=0
    )

    # Spectral Centroid
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=signal,
            sr=sample_rate
        ).T,
        axis=0
    )

    # Combine all features into one vector
    features = np.hstack([
        mfcc,
        chroma,
        mel,
        zcr,
        rms,
        spectral_centroid
    ])

    return features