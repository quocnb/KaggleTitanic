import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


def tune(x, y):
    params = {
        'reg_alpha': np.arange(0.1, 1, 1e-1).tolist(),
        'max_depth': range(1, 10, 1),
        'min_child_weight': range(1, 10, 1),
        'gamma': np.arange(0.1, 1, 1e-1).tolist(),
        'subsample': np.arange(0.1, 1, 1e-1).tolist(),
        'colsample_bytree': np.arange(0.1, 1, 1e-1).tolist()
    }
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
