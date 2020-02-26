import os
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import matplotlib.pyplot as plt
from data_visualization import sns, cfg
from sklearn.metrics import mean_squared_error
int_cols = cfg.int_cols


def nan_imputer(data: pd.DataFrame, tcol: str, window: int = 24, verbose=cfg.verbose):
    df = data.copy()
    init_cols = df.columns.tolist()
    if verbose:
        st = (11, '2017-07-01 00:00:00')
        end = (11, '2018-07-01 00:00:00')
        fig, ax = plt.subplots(1, 1, figsize=(14, 6), dpi=100)
        df_for_plot = df.loc[st:end]
        df_for_plot.set_index('timestamp')[[
            f'{tcol}'
        ]].plot(ax=ax)
        fig.show()
        fig.savefig(cfg.eda_dir + "/{}_before_filling_missing_sequence.png".format(tcol))
    df['rolling_back'] = df.groupby(by='site_id')[tcol] \
        .rolling(window=window, min_periods=1).mean().interpolate().values

    # # reversed rolling
    tmp = df.iloc[::-1]
    df['rolling_forw'] = tmp.groupby(by='site_id')[tcol] \
        .rolling(window=window, min_periods=1).mean().interpolate().values

    # rolling mean for same hour of the day
    df['rolling_back_h'] = df.groupby(by=['site_id', 'hour'])[tcol] \
        .rolling(window=3, min_periods=1).mean().interpolate().values

    df['rolling_back_h_f'] = df.iloc[::-1].groupby(by=['site_id', 'hour'])[tcol] \
        .rolling(window=3, min_periods=1).mean().interpolate().values # [::-1] start from end in python

    features = [
        'site_id', 'hour', 'month', 'dayofyear', 'weekofyear', 'year',
        'rolling_back',
        'rolling_forw',
        'rolling_back_h',
        'rolling_back_h_f', 'season', f'{tcol}_mean_lag3', f'{tcol}_std_lag3',f'{tcol}_mean_lag72', f'{tcol}_std_lag72'
    ]
    tr_idx = ~(df[tcol].isnull()) & (df[tcol] != 0)  # zero in feature column means sites that didnt collect the feature
    val_idx = df[tcol].isnull()
    print(f'training model for col "{tcol}"...')
    if tcol in cfg.int_cols:
        lgbm = lgb.LGBMClassifier(
                                    learning_rate=0.05,
                                    objective='rmse',
                                    n_estimators=350,
                                    num_threads=os.cpu_count(),
                                    num_leaves=256,
                                    max_depth=8,
                                    subsample=0.8,
                                    min_child_samples=50,
                                    random_state=42)
    else:
        lgbm = lgb.LGBMRegressor(
                                    learning_rate=0.05,
                                    objective='rmse',
                                    n_estimators=350,
                                    num_threads=os.cpu_count(),
                                    num_leaves=256,
                                    max_depth=8,
                                    subsample=0.8,
                                    min_child_samples=50,
                                    random_state=42)

    lgbm.fit(
        X=df.loc[tr_idx, features],
        y=df.loc[tr_idx, tcol],
        categorical_feature=['site_id', 'year'], verbose=verbose
    )
    training_set_prediction = lgbm.predict(df.loc[tr_idx, features])
    training_rmse = mean_squared_error(df.loc[tr_idx, tcol], training_set_prediction) ** 0.5
    print("training rmse : \t {}".format(training_rmse))
    df[f'{tcol}_restored'] = np.nan
    tmp = lgbm.predict(df.loc[val_idx, features])
    df.loc[val_idx, f'{tcol}_restored'] = tmp
    if verbose:
        # check from what features our imputer learned the most

        # lgb.plot_importance(lgbm)
        # plt.show()

        df_for_plot = df.loc[:, init_cols + [f'{tcol}_restored']]
        df_for_plot = df_for_plot.loc[st:end]

        fig, ax = plt.subplots(1, 1, figsize=(14, 6), dpi=100)
        df_for_plot.set_index('timestamp')[[
            f'{tcol}',
            f'{tcol}_restored'
        ]].plot(ax=ax)
        fig.show()
        fig.savefig(cfg.eda_dir + "/{}_filling_missing_sequence.png".format(tcol))

    df.loc[val_idx, f'{tcol}'] = df.loc[val_idx, f'{tcol}_restored'].values
    df = df.loc[:, init_cols]
    return df
