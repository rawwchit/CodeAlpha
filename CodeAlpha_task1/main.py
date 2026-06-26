import pandas as pd
from src.FE import create_features
from src.preprocessing import preprocess
from src.models import train_logistic
from src.models import train_dt
from src.models import train_rf
from src.evaluate import evaluate_model

df = pd.read_csv("data/credit_data.csv")

#Data exploration
print(df.head()) 
print(df.shape) 
# print(df.isnull().sum())
# print(df.columns.tolist())

df = create_features(df)

x_train,x_test, y_train, y_test, scaler = preprocess(df)

# Understanding Target
count = df['DEFAULT'].value_counts()
print(count)
print("\nPercentage : ",df['DEFAULT'].value_counts(normalize=True)*100)
ratio = (count[0]/count[1])
print("\nRatio : ",ratio)

print("\nLOGISTIC REGRESSION")
model = train_logistic(x_train,y_train)
evaluate_model(model,x_test,y_test)

print("\nDECISION TREE")
dt = train_dt(x_train, y_train)
evaluate_model(dt,x_test,y_test)

print("\nRANDOM FOREST")
rf = train_rf(x_train, y_train)
evaluate_model(rf,x_test,y_test)
