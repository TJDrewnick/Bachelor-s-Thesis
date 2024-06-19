import lightgbm as lgb
import numpy as np
import xgboost as xgb
import shap
import joblib as jbl
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import uniform, randint

from utils import ml_parameters
from utils import settings as s


def fit_lgb(X_train: np.array, y_train: np.array, parameters: dict) -> lgb.LGBMRegressor:
    best_parameters = parameters

    if s.ml['random search gbt_lgb']:
        param_distributions = {
            'num_leaves': randint(10, 100),
            'learning_rate': uniform(0.01, 0.5),
            'max_depth': randint(3, 12),
            'n_estimators': randint(100, 1000),
            'min_child_samples': randint(20, 80),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1),
        }

        random_search = RandomizedSearchCV(
            estimator=lgb.LGBMRegressor(random_state=42, boosting_type='gbdt'),
            param_distributions=param_distributions,
            n_iter=s.ml['random search iterations'],
            cv=5,
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(X_train, y_train)
        best_parameters = random_search.best_estimator_.get_params()

    # fit model
    model = lgb.LGBMRegressor(
        random_state=42,
        boosting_type='gbdt',
        num_leaves=best_parameters['num_leaves'],
        learning_rate=best_parameters['learning_rate'],
        max_depth=best_parameters['max_depth'],
        n_estimators=best_parameters['n_estimators'],
        min_child_samples=best_parameters['min_child_samples'],
        reg_alpha=best_parameters['reg_alpha'],
        reg_lambda=best_parameters['reg_lambda']
    )
    model.fit(X_train, y_train)
    return model


def fit_xgb(X_train: np.array, y_train: np.array, parameters: dict) -> xgb.XGBRegressor:
    best_parameters = parameters

    if s.ml['random search gbt_xgb_squarederror']:
        param_distributions = {
            'eta': uniform(0.01, 0.5),
            'max_depth': randint(3, 12),
            'n_estimators': randint(100, 1000),
            'min_child_weight': randint(1, 30),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1),
        }

        random_search = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(random_state=42),
            param_distributions=param_distributions,
            n_iter=s.ml['random search iterations'],
            cv=5,
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(X_train, y_train)
        best_parameters = random_search.best_estimator_.get_params()

    model = xgb.XGBRegressor(
        random_state=42,
        eta=best_parameters['eta'],
        max_depth=best_parameters['max_depth'],
        n_estimators=best_parameters['n_estimators'],
        min_child_weight=best_parameters['min_child_weight'],
        reg_alpha=best_parameters['reg_alpha'],
        reg_lambda=best_parameters['reg_lambda']
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_true: np.array, y_pred: np.array,  model_type: str) -> None:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    with open('../results/ml_v1/model_errors.txt', mode='a') as file:
        file.write(f'\n{model_type}\n')
        file.write(f'Mean Squared Error: {mse}\n')
        file.write(f'Mean Absolute Error: {mae}\n')
        file.write(f'Mean Absolute Percentage Error: {mape}\n')
        file.write(f'r2-Score: {r2}\n')


def save_model_parameters(model, model_type: str) -> None:
    params_model = model.get_params()

    with open('../results/ml_v1/model_parameters.txt', mode='a') as file:
        file.write(f'\n{model_type}\n')
        for key, value in params_model.items():
            if value is not None:
                file.write(f'{key}: {value}\n')


# clear model parameters and errors files
open('../results/ml_v1/model_parameters.txt', 'w').close()
open('../results/ml_v1/model_errors.txt', 'w').close()

for area in ['AUS', 'CE']:
    for datatype in ['detrended', 'original']:

        y_complete = pd.DataFrame()
        y_complete_all = pd.DataFrame()

        data = pd.read_hdf(f'../results/prepared_features/{area}_{datatype}_ml.h5')

        feature_columns = [column for column in data.columns if column not in ['drift', 'diffusion']]
        X = data[feature_columns]
        for target in ['drift', 'diffusion']:
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=s.ml['test size'], random_state=42
            )

            # fit models
            lgb_model = fit_lgb(
                X_train, y_train,
                ml_parameters.parameters[area][datatype][target]['lgb'],
            )
            xgb_model = fit_xgb(
                X_train, y_train,
                ml_parameters.parameters[area][datatype][target]['xgb'],
            )

            # predict models
            y_pred_lgb = lgb_model.predict(X_test)
            y_pred_xgb = xgb_model.predict(X_test)

            # evaluate models
            model_description = f'{area}: {datatype.capitalize()} {target.capitalize()}'
            evaluate_model(y_test, y_pred_lgb,  f'{model_description} (LightGMB)')
            evaluate_model(y_test, y_pred_xgb, f'{model_description} (XGBoost)')

            # model parameters
            save_model_parameters(lgb_model, f'LightGMB: {model_description}')
            save_model_parameters(xgb_model, f'XGBoost: {model_description}')

            # shap feature importance
            lgb_explainer = shap.Explainer(lgb_model)
            xgb_explainer = shap.Explainer(xgb_model)

            # save shap explainers
            with open(
                    f'../results/ml_v1/explainers/{area}_{datatype}_{target}_gbt_lgb_shap_values', 'wb'
            ) as file:
                jbl.dump(lgb_explainer, file)
            with open(
                    f'../results/ml_v1/explainers/{area}_{datatype}_{target}_gbt_xgb_squarederror_shap_values', 'wb'
            ) as file:
                jbl.dump(xgb_explainer, file)

            # store y values for box plot
            y_complete[f'{target}_true'] = y_test
            y_complete[f'{target}_gbt_lgb'] = y_pred_lgb
            y_complete[f'{target}_gbt_xgb_squarederror'] = y_pred_xgb

            # store y values for all the data
            y_complete_all[f'{target}_true_all'] = y
            y_complete_all[f'{target}_gbt_lgb_all'] = lgb_model.predict(X)
            y_complete_all[f'{target}_gbt_xgb_squarederror_all'] = xgb_model.predict(X)

        # save all predictions and true values
        y_complete.to_hdf(f'../results/ml_v1/predictions/{area}_{datatype}_predictions.h5', key='df', mode='w')
        y_complete_all.to_hdf(f'../results/ml_v1/predictions/{area}_{datatype}_all_predictions.h5', key='df', mode='w')
