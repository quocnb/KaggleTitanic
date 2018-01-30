import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble, gaussian_process, linear_model, naive_bayes
from sklearn import neighbors, svm, tree, model_selection, discriminant_analysis
from sklearn.metrics import roc_curve, precision_score, recall_score, auc
import xgboost as xgb
import numpy as np

# Load data
df_train = pd.read_csv('./input/train.csv')
df_test = pd.read_csv('./input/test.csv')
df_total = [df_train, df_test]

test_passenger_id = df_test['PassengerId']

for df in df_total:
    df['Salutation'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

for df in df_total:
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

for df in df_total:
    df.drop(
        ['PassengerId', 'Name', 'Ticket', 'Cabin'],
        axis=1,
        inplace=True
    )


def fill_age(dataframe, inplace=True):
    length = len(dataframe.groupby('Salutation'))
    for index in range(length - 1):
        median_age = dataframe[dataframe['Salutation'] == index]['Age'].median()
        dataframe['Age'].fillna(median_age, inplace=inplace)


for df in df_total:
    fill_age(df)


def fill_embarked(dataframe, inplace=True):
    dataframe['Embarked'].fillna('C', inplace=inplace)


for df in df_total:
    fill_embarked(df)


def fill_fare(dataframe, inplace=True):
    condition = (dataframe['Pclass'] == 3) \
                & (dataframe['Embarked'] == 'S') \
                & (dataframe['Salutation'] == 0)
    median = dataframe[condition]['Fare'].median()
    dataframe['Fare'].fillna(median, inplace=inplace)


for df in df_total:
    fill_fare(df)

for dataset in df_total:
    dataset.loc[dataset["Age"] <= 9, "Age"] = 0
    dataset.loc[(dataset["Age"] > 9) & (dataset["Age"] <= 19), "Age"] = 1
    dataset.loc[(dataset["Age"] > 19) & (dataset["Age"] <= 29), "Age"] = 2
    dataset.loc[(dataset["Age"] > 29) & (dataset["Age"] <= 39), "Age"] = 3
    dataset.loc[(dataset["Age"] > 29) & (dataset["Age"] <= 39), "Age"] = 3
    dataset.loc[dataset["Age"] > 39, "Age"] = 4

for dataset in df_total:
    dataset.loc[dataset["Fare"] <= 7.75, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 7.75) & (dataset["Fare"] <= 7.91), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 9.841), "Fare"] = 2
    dataset.loc[(dataset["Fare"] > 9.841) & (dataset["Fare"] <= 14.454), "Fare"] = 3
    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 24.479), "Fare"] = 4
    dataset.loc[(dataset["Fare"] > 24.479) & (dataset["Fare"] <= 31), "Fare"] = 5
    dataset.loc[(dataset["Fare"] > 31) & (dataset["Fare"] <= 69.487), "Fare"] = 6
    dataset.loc[dataset["Fare"] > 69.487, "Fare"] = 7

for dataset in df_total:
    dataset['Sex'] = pd.factorize(dataset['Sex'])[0]
    dataset['Embarked'] = pd.factorize(dataset['Embarked'])[0]

x = df_train.drop('Survived', axis=1).values
y = df_train['Survived'].values


def compare_algorithm(data, target):
    x_train, x_cross, y_train, y_cross = train_test_split(data, target)
    MLA = [
        # Ensemble Methods
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),

        # Gaussian Processes
        gaussian_process.GaussianProcessClassifier(),

        # GLM
        linear_model.LogisticRegressionCV(),
        linear_model.PassiveAggressiveClassifier(max_iter=1000, tol=0.001),
        linear_model.RidgeClassifierCV(),
        linear_model.SGDClassifier(max_iter=1000, tol=0.001),
        linear_model.Perceptron(max_iter=1000, tol=0.001),

        # Navies Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),

        # Nearest Neighbor
        neighbors.KNeighborsClassifier(),

        # SVM
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        svm.LinearSVC(),

        # Trees
        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),

        # Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),

        # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
        xgb.XGBClassifier()
    ]
    MLA_columns = []
    MLA_compare = pd.DataFrame(columns=MLA_columns)

    row_index = 0
    for alg in MLA:
        predicted = alg.fit(x_train, y_train).predict(x_cross)
        fp, tp, th = roc_curve(y_cross, predicted)
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(x_train, y_train), 4)
        MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(x_cross, y_cross), 4)
        MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(y_cross, predicted)
        MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(y_cross, predicted)
        MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)
        row_index = row_index + 1

    MLA_compare.sort_values(
        by=['MLA Test Accuracy'],
        ascending=False,
        inplace=True
    )
    print(MLA_compare)


# Find best algorithm
compare_algorithm(x, y)

algorithm = ensemble.RandomForestClassifier()
param_grid = {
    'n_estimators': np.arange(2, 20, 2).tolist(),
    'criterion': ['gini'],
    'class_weight': ['balanced', None],
    'max_depth': np.arange(2, 10, 2).tolist(),
    'max_features': ['log2', 'auto'],
    'max_leaf_nodes': np.arange(2, 8, 2).tolist(),
    'n_jobs': np.arange(2, 20, 2).tolist(),
}
tune_model = model_selection.GridSearchCV(
    algorithm,
    param_grid=param_grid,
    scoring='roc_auc'
)
tune_model.fit(x, y)
print('Tuning Parameters: ', tune_model.best_params_)
print("Tuning Score: ", tune_model.score(x, y))
print('-'*50)

print(df_test.isnull().sum())
x_test = df_test.values
y_test = tune_model.predict(x_test)

output = pd.DataFrame(
    {
        'PassengerId': test_passenger_id,
        'Survived': y_test
    }
)
output.head()
output.to_csv('./output/submission.csv', index=False)

#Tuning Parameters:  {'class_weight': None, 'criterion': 'gini', 'max_depth': 4, 'max_features': 'log2', 'max_leaf_nodes': 6, 'n_estimators': 8, 'n_jobs': 14}
#0.78468