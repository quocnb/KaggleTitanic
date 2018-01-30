from PreprocessTitanicData import pre_processing
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

df_train = pd.read_csv('./input/train.csv')
df_test = pd.read_csv('./input/test.csv')
df_total = [df_train, df_test]

test_passenger_id = df_test['PassengerId']

for df in df_total:
    pre_processing(df)

x = df_train.drop('Survived', axis=1).values
y = df_train['Survived'].values

params = {
        'reg_alpha': np.arange(0.1, 1, 2e-1).tolist(),
        'max_depth': range(1, 10, 2),
        'min_child_weight': range(1, 10, 2),
        'gamma': np.arange(0.1, 1, 2e-1).tolist(),
        'subsample': np.arange(0.1, 1, 2e-1).tolist(),
        'colsample_bytree': np.arange(0.1, 1, 2e-1).tolist()
    }
# Best params =  {'colsample_bytree': 0.9000000000000001, 'gamma': 0.30000000000000004, 'max_depth': 9, 'min_child_weight': 1, 'reg_alpha': 0.1, 'subsample': 0.5000000000000001}
# With score: 0.8742624163927456
# Kaggle score: 0.79425
algorithm = XGBClassifier(
    learning_rate=0.01,
    n_estimators=100,
    max_depth=6,
    min_child_weight=4,
    gamma=0.0,
    subsample=0.8,
    colsample_bytree=0.75,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27,
    reg_alpha=0.05
)
gsearch = GridSearchCV(
    estimator=algorithm,
    param_grid=params,
    scoring='roc_auc',
    n_jobs=10,
    iid=False,
    cv=5
)
gsearch.fit(x, y)
print('Best params = ', gsearch.best_params_)
print('With score:', gsearch.best_score_)

# algorithm = XGBClassifier(
#             learning_rate=0.1,
#             n_estimators=140,
#             max_depth=6,
#             min_child_weight=4,
#             gamma=0.0,
#             subsample=0.8,
#             colsample_bytree=0.75,
#             objective='binary:logistic',
#             nthread=4,
#             scale_pos_weight=1,
#             seed=27,
#             reg_alpha=0.05
#         )
# algorithm.fit(x, y)
# print('Score = ', algorithm.score(x, y))
x_test = df_test.values
y_test = gsearch.predict(x_test)
output = pd.DataFrame(
    {
        'PassengerId': test_passenger_id,
        'Survived': y_test
    }
)
output.head()
output.to_csv('./output/submission_xgboost.csv', index=False)
