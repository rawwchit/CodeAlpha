# Data Preprocessing 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    x = df.drop("DEFAULT", axis = 1)
    y = df["DEFAULT"]

    # Remove identifier columns if present
    x = x.drop(columns=['CUST_ID'], errors='ignore')
    x = x.drop(columns=['ID'], errors='ignore')

    # Convert categorical variables into numerical form
    x = pd.get_dummies(x, drop_first=True)
    
    # Replace infinities with NaN
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill missing values
    x.fillna(0, inplace=True)
    # Data quality check
    print("\nData Quality Check")
    print("NaNs :", x.isna().sum().sum())
    print("Infs :",np.isinf(x.select_dtypes(include=np.number)).sum().sum()
)
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    return (
        x_train,
        x_test,
        y_train,
        y_test,
        scaler
        )
