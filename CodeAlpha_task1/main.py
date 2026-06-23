import pandas as pd
df = pd.read_csv("credit_data.csv")
#Data exploration
print(df.head()) 
print(df.shape) 
# print(df.isnull().sum())
# print(df.columns.tolist())

# Understanding Target
count = df['DEFAULT'].value_counts()
print(count)
print("\nPercentage : ",df['DEFAULT'].value_counts(normalize=True)*100)
ratio = (count[0]/count[1])
print("\nRatio : ",ratio)

# Feature Engineering
df["DTI"] = df["DEBT"]/df["INCOME"]
df["SAVINGS_RATIO"] = df["SAVINGS"]/df["INCOME"]
df["SPEND_RATIO"] = df["T_EXPENDITURE_12"]/df["INCOME"]
import numpy as np
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)


# Data Preparation
x = df.drop("DEFAULT", axis = 1)
y = df["DEFAULT"]
# Check categorical columns
print("\nCategorical Columns:")
# Remove identifier columns if present
x = x.drop(columns=['CUST_ID'], errors='ignore')
x = x.drop(columns=['ID'], errors='ignore')
# Convert categorical variables into numerical form
x = pd.get_dummies(x, drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("NaNs :", x.isna().sum().sum())
print("Infs :", np.isinf(x.select_dtypes(include=np.number)).sum().sum())

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

# First Model Training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=2000)
model.fit(x_train,y_train)
print(x.select_dtypes(include=['object','string']).columns.tolist())