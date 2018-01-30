import pandas as pd


def pre_processing(df):
    extract_name(df)
    drop_unnecessary_columns(df)
    fill_nan(df)
    extract_age(df)
    extract_fare(df)
    process_sex(df)
    process_embarked(df)
    
    
def extract_name(df):
    df['Salutation'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Salutation'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Rare',
        inplace=True
    )
    df['Salutation'].replace(
        ['Mlle', 'Ms'],
        'Miss',
        inplace=True
    )
    df['Salutation'].replace(
        'Mme',
        'Mrs',
        inplace=True
    )
    df['Salutation'] = pd.factorize(df['Salutation'])[0]


def drop_unnecessary_columns(df):
    df.drop(
        ['PassengerId', 'Name', 'Ticket', 'Cabin'],
        axis=1,
        inplace=True
    )


def fill_nan(df):
    fill_age(df)
    fill_embarked(df)
    fill_fare(df)


def extract_age(df):
    age = df['Age']
    df.loc[age <= 9, "Age"] = 0
    df.loc[(age > 9) & (age <= 19), "Age"] = 1
    df.loc[(age > 19) & (age <= 29), "Age"] = 2
    df.loc[(age > 29) & (age <= 39), "Age"] = 3
    df.loc[(age > 29) & (age <= 39), "Age"] = 3
    df.loc[age > 39, "Age"] = 4


def extract_fare(df):
    fare = df['Fare']
    df.loc[fare <= 7.75, "Fare"] = 0
    df.loc[(fare > 7.75) & (fare <= 7.91), "Fare"] = 1
    df.loc[(fare > 7.91) & (fare <= 9.841), "Fare"] = 2
    df.loc[(fare > 9.841) & (fare <= 14.454), "Fare"] = 3
    df.loc[(fare > 14.454) & (fare <= 24.479), "Fare"] = 4
    df.loc[(fare > 24.479) & (fare <= 31), "Fare"] = 5
    df.loc[(fare > 31) & (fare <= 69.487), "Fare"] = 6
    df.loc[fare > 69.487, "Fare"] = 7


def process_sex(df):
    df['Sex'] = pd.factorize(df['Sex'])[0]


def process_embarked(df):
    df['Embarked'] = pd.factorize(df['Embarked'])[0]


def fill_age(df):
    length = len(df.groupby('Salutation'))
    for index in range(length - 1):
        median_age = df[df['Salutation'] == index]['Age'].median()
        df['Age'].fillna(median_age, inplace=True)


def fill_embarked(df):
    df['Embarked'].fillna('C', inplace=True)


def fill_fare(df):
    condition = (df['Pclass'] == 3) & (df['Embarked'] == 'S') & (df['Salutation'] == 0)
    median = df[condition]['Fare'].median()
    df['Fare'].fillna(median, inplace=True)
