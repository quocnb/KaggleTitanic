import pandas as pd
from PreprocessTitanicData import pre_processing
from sklearn.linear_model import LogisticRegressionCV

df_train = pd.read_csv('./input/train.csv')
df_test = pd.read_csv('./input/test.csv')
df_total = [df_train, df_test]

for df in df_total:
    pre_processing(df)

lr = LogisticRegressionCV(

)