import pandas as pd
from sklearn.model_selection import train_test_split, KFold

df = pd.read_csv("E:/Dataset/morph2-224.csv")
# test = df.sample(frac=0.2, random_state=2, replace=False)
# train = test

train, test =  train_test_split(df, test_size=0.2, random_state=2)
train.to_csv("E:/Dataset/morph2-224_train.csv", index=False)
test.to_csv("E:/Dataset/morph2-224_test.csv", index=False)
