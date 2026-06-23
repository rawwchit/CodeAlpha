import pandas as pd
df = pd.read_csv("credit_data.csv")
#Data exploration
print(df.head()) 
print(df.shape) 
# print(df.isnull().sum())

# Understanding Target
count = df['DEFAULT'].value_counts()
print(count)
print("\nPercentage : ",df['DEFAULT'].value_counts(normalize=True)*100)
ratio = (count[0]/count[1])
print("\nRatio : ",ratio)

# Feature Engineering
df["DTI"] = df["DEBT"]/df["INCOME"]
df["SAVINGS_RATIO"] = df["SAVINGS"]/df["INCOME"]
df["SPEND_RATIO"] = df["TOTAL_SPEND"]/df["INCOME"]

# Data Preparation
x = df.drop("DEFAULT", axis = 1)
y = df.drop("DEFAULT")
#split
from sklearn.model_selection import train_test_split
x_train, x-test, y_train, y_test = train_test_split(
    x,
    y,
    test_code = 0.2,
    random_state = 42,
)

# First Model Training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)