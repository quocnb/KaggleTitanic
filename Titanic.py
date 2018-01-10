import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def process_dataframe(df):
    # Drop unnecessary column
    df.drop(['PassengerId', 'Ticket', 'Name'], axis=1, inplace=True)

    # Fill NaN value
    df.Age.fillna(df.Age.median(), inplace=True)
    df.Embarked.fillna(df.Embarked.mode()[0], inplace=True)
    df.Fare.fillna(df.Fare.median(), inplace=True)
    # Drop NaN
    df.drop('Cabin', axis=1, inplace=True)

    # Convert string to float by using LabelEncoder
    lb = LabelEncoder()
    df.Sex = lb.fit_transform(df.Sex)
    df = pd.get_dummies(df, columns=['Embarked'])
    if 'Survived' in df.columns:
        y = df.Survived.values
        X = df.drop('Survived', axis=1).values
        return X, y
    else:
        X = df.values
        return X


df_train = pd.read_csv('./input/train.csv')
X_train, y_train = process_dataframe(df_train)

forest = RandomForestClassifier(25)
forest.fit(X_train, y_train)
print('Training score = ', forest.score(X_train, y_train))

ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
lr = LogisticRegression(penalty='l1', C=0.01, max_iter=1000)
lr.fit(X_train_std, y_train)
print('Training score = (lr)', lr.score(X_train_std, y_train))

df_test = pd.read_csv('./input/test.csv')
pass_id = df_test['PassengerId']
X_test = process_dataframe(df_test)
X_test_std = ss.fit_transform(X_test)
y_pred = lr.predict(X_test_std)
output = pd.DataFrame({'PassengerId': pass_id,
                       'Survived': y_pred})
output.head()
output.to_csv('gender_submission.csv', index=False)
