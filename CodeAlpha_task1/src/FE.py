# Feature Engineering
import numpy as np
def create_features(df):
    df["DTI"] = df["DEBT"]/df["INCOME"]
    df["SAVINGS_RATIO"] = df["SAVINGS"]/df["INCOME"]
    df["SPEND_RATIO"] = df["T_EXPENDITURE_12"]/df["INCOME"]
    # replacing infinity
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df