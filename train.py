from data_manipulation import *
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pickle
from tqdm import tqdm
import optuna
from optuna import Trial
import gc
optuna.logging.set_verbosity(optuna.logging.WARNING)

def build_result_line(title, detail, dataset,  model_name, rmse):
    return {'title': title, 'detail': detail, 'dataset': dataset, 'model_name': model_name, 'rmse': rmse}

#
# def lgbm_model(train_X_df, train_Y_df, folds):
#     print("-------------Training LGBM model with K-Fold -------------")
#
#     categoricals = ["site_id", "building_id", "primary_use", "meter", "wind_direction", 'cloud_coverage', 'season']
#     categoricals_final = []
#     for col in train_X_df.columns:
#         if col in categoricals:
#             categoricals_final.append(col)
#
#     rmse = []
#     seed = 666
#     shuffle = False
#     kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)
#     models = []
#     params = {
#         'boosting_type': 'gbdt',  # default for decision trees
#         'objective': 'regression',
#         'metric': {'rmse'},
#         'subsample_freq': 1,
#         'learning_rate': 0.3,
#         'bagging_freq': 5,  # every 5 iteration preform sampling with replacement of the data
#         'num_leaves': 350,  # max leaves in each decision tree
#         'feature_fraction': 0.9,  # randomly select part of the features for each the tree
#         'lambda_l1': 1,  # regularizaion weigh
#         'lambda_l2': 1
#     }
#
#     for train_index, val_index in kf.split(train_X_df):
#         train_X = train_X_df.iloc[train_index]
#         val_X = train_X_df.iloc[val_index]
#         train_y = train_Y_df.iloc[train_index]
#         val_y = train_Y_df.iloc[val_index]
#         lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals_final)
#         lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals_final)
#         gbm = lgb.train(params,
#                         lgb_train,
#                         num_boost_round=500,
#                         valid_sets=(lgb_train, lgb_eval),
#                         early_stopping_rounds=50,
#                         verbose_eval=50)
#         models.append(gbm)
#         val_predict_value = gbm.predict(val_X)
#         rmse.append(mean_squared_error(val_y, val_predict_value) ** 0.5)
#     if cfg.verbose:
#         # %%see which variables are the most relevant
#         feature_imp = pd.DataFrame(sorted(zip(gbm.feature_importance(), gbm.feature_name()), reverse=True),
#                                    columns=['Value', 'Feature'])
#         feature_imp.to_pickle(path=cfg.feature_df)
#         plt.figure(figsize=(10, 5))
#         sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
#         plt.title('LightGBM FEATURES')
#         plt.tight_layout()
#         plt.show()
#         print("")
#     rmse = np.mean(rmse)
#     return models, rmse
#
#
# def training_per_meter(X, Y):
#     # folds = 5
#     models_per_meter_id = []
#     rmse_mean = []
#     categoricals = ["site_id", "building_id", "primary_use", "meter", "wind_direction", 'cloud_coverage', 'season']
#     categoricals_final = []
#     for col in X.columns:
#         if col in categoricals:
#             categoricals_final.append(col)
#
#     for meter_id in X['meter'].value_counts(dropna=False).index.to_list():
#         print("--------- training for meter {} ---------".format(meter_id))
#         indices_by_meter_id = list(X[X['meter'] == meter_id].index)
#         train_X_meter = X.loc[indices_by_meter_id]
#         train_y_meter = Y.loc[indices_by_meter_id]
#         train_X_meter.drop('meter', inplace=True, axis='columns')
#         models, rmse = lgbm_model(train_X_meter, train_y_meter, cfg.folds)
#         models_per_meter_id.append(models)
#         rmse_mean.append(rmse)
#     rmse = np.mean(rmse_mean)
#     print("RMSE for validation set : {}".format(rmse))
#     print("------------- Saving models -------------")
#     for j, models in enumerate(models_per_meter_id):
#         for i, model in enumerate(models):
#             with open(cfg.model_dir + 'meter_{}_model_{}.pkl'.format(j, i), 'wb') as fout:
#                 pickle.dump(model, fout)
#     return rmse
#

def lgbm_with_optuna_opt_hp(trial, train, val, seed=None, cat_features=None, num_rounds=1000, prune=True, verbose=-1):
    """Train Light GBM model"""
    X_train, y_train = train
    X_valid, y_valid = val
    n_jobs = os.cpu_count() if prune else os.cpu_count()
    params = {
            'num_leaves': trial.suggest_int('num_leaves', 2, 350),
            'objective': 'regression',
            'learning_rate': trial.suggest_uniform('learning_rate', 1e-4, 1.0),
            "boosting": "gbdt",
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            "bagging_freq": 5,
            "bagging_fraction": trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
            "feature_fraction": trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            "metric": {'rmse'},
            "verbose": verbose,
            "n_jobs": n_jobs}
    params['seed'] = seed
    d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, params={'verbose':verbose})
    d_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_features, params={'verbose': verbose})
    watchlist = [d_train, d_valid]
    print('training LGB:')
    if prune:
        early_stop = 50
        verbose_eval = 50
        # Add a callback for pruning.
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'rmse', valid_name='valid_1')

        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop,
                          callbacks=[pruning_callback])
    else:
        early_stop = 50
        verbose_eval = 50
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)

    # predictions
    print('best_score', model.best_score)
    log = {'train/rmse': model.best_score['training']['rmse'],
           'valid/rmse': model.best_score['valid_1']['rmse']}
    return model, log


def objective_with_prune(trial: Trial, X_train, y_train):
    categoricals = ["site_id", "building_id", "primary_use", "meter", "wind_direction", 'cloud_coverage', 'season']
    category_cols = []
    for col in X_train.columns:
        if col in categoricals:
            category_cols.append(col)
    seed = 666
    shuffle = False
    kf = KFold(n_splits=5, shuffle=shuffle, random_state=seed)
    valid_score = 0
    for train_idx, valid_idx in kf.split(X_train, y_train):
        train_X = X_train.iloc[train_idx]
        val_X = X_train.iloc[valid_idx]
        train_y = y_train.iloc[train_idx]
        val_y = y_train.iloc[valid_idx]
        train_data = (train_X, train_y)
        valid_data = (val_X, val_y)
        print('train', len(train_idx), 'valid', len(valid_idx))
        _, log = lgbm_with_optuna_opt_hp(trial, train_data, valid_data, cat_features=category_cols,
                                          num_rounds=500)
        valid_score += log["valid/rmse"]
        break
    return valid_score


def get_important_features_dataframes():
    features_df = []
    for i in range(4):
        with open(cfg.feature_dir + 'meter_{}.pkl'.format(i), 'rb') as fout:
            feature_dataframe = pickle.load(fout)
            features_df.append(feature_dataframe)
    return features_df


def use_optuna_find_best_parameters(X:pd.DataFrame, Y:pd.DataFrame, objective_fn=objective_with_prune, sample=True, use_important_features=False):

    if sample:
        # Only use 100000 data
        X_sample = X.sample(100000)
        indices = list(X_sample.index)
        Y_sample = Y.loc[indices]
    else:
        X_sample = X
        Y_sample = Y
    if use_important_features: # need to create this files in the first place using the plotting_train_prediction function
        important_features_dataframes = get_important_features_dataframes()
        important_features_per_meter = [dataframe[dataframe['Value'] > 3]['Feature'] for dataframe in important_features_dataframes]
    study_per_meter_id = []
    for meter_id in X['meter'].value_counts(dropna=False).index.to_list():

        print("--------- finding best parameters for meter {} ---------".format(meter_id))
        indices_by_meter_id = list(X_sample[X_sample['meter'] == meter_id].index)
        train_X_meter = X_sample.loc[indices_by_meter_id]
        train_y_meter = Y_sample.loc[indices_by_meter_id]
        train_X_meter.drop('meter', inplace=True, axis='columns')
        if use_important_features:
            important_feat = important_features_per_meter[meter_id]
            train_X_meter = train_X_meter[important_feat]
        study = optuna.create_study(pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=4,
                                                                                  min_early_stopping_rate=1))
        study.optimize(lambda trial: objective_fn(trial, train_X_meter,
                                                          train_y_meter),
                       timeout=cfg.timeout,  n_jobs=os.cpu_count())
        gc.collect()
        study_per_meter_id.append(study)
        print(
            f'Executed {len(study.trials)} trials, best score {study.best_value} with best_params {study.best_params}')

    return study_per_meter_id



def objective(trial: Trial, X_train, y_train):
    categoricals = ["site_id", "building_id", "primary_use", "meter", "wind_direction", 'cloud_coverage', 'season']
    category_cols = []
    for col in X_train.columns:
        if col in categoricals:
            category_cols.append(col)
    # seed = 666
    # shuffle = False
    kf = KFold(n_splits=cfg.folds)
    gc.collect()
    models0 = []
    valid_score = 0
    for train_idx, valid_idx in kf.split(X_train, y_train):
        train_X = X_train.iloc[train_idx]
        val_X = X_train.iloc[valid_idx]
        train_y = y_train.iloc[train_idx]
        val_y = y_train.iloc[valid_idx]
        train_data = (train_X, train_y)
        valid_data = (val_X, val_y)
        print('train', len(train_idx), 'valid', len(valid_idx))
        model, log = lgbm_with_optuna_opt_hp(trial, train_data, valid_data, cat_features=category_cols,
                                                           num_rounds=1000, prune=False, verbose=0)
        models0.append(model)
        gc.collect()
        valid_score += log["valid/rmse"]
    valid_score /= len(models0)
    return valid_score, models0

def training_per_meter_with_optuna(X, Y, best_params, use_important_features=False):
    models_per_meter_id = []
    rmse_mean = []
    categoricals = ["site_id", "building_id", "primary_use", "meter", "wind_direction", 'cloud_coverage', 'season']
    categoricals_final = []
    for col in X.columns:
        if col in categoricals:
            categoricals_final.append(col)
    if use_important_features:
        important_features_dataframes = get_important_features_dataframes()
        important_features_per_meter = [dataframe[dataframe['Value'] > 3]['Feature'] for dataframe in important_features_dataframes]

    for meter_id in X['meter'].value_counts(dropna=False).index.to_list():
        important_feat = important_features_per_meter[meter_id]
        best_params_per_meter = best_params[meter_id]
        print("--------- training for meter {} ---------".format(meter_id))
        indices_by_meter_id = list(X[X['meter'] == meter_id].index)
        train_X_meter = X.loc[indices_by_meter_id]
        train_y_meter = Y.loc[indices_by_meter_id]
        train_X_meter.drop('meter', inplace=True, axis='columns')
        train_X_meter = train_X_meter[important_feat]
        rmse, models = objective(optuna.trial.FixedTrial(best_params_per_meter.best_params),
                                             X_train=train_X_meter, y_train=train_y_meter)
        gc.collect()
        models_per_meter_id.append(models)
        rmse_mean.append(rmse)
    rmse = np.mean(rmse_mean)
    print("RMSE for validation set : {}".format(rmse))
    print("------------- Saving models -------------")
    for j, models in enumerate(models_per_meter_id):
        for i, model in enumerate(models):
            with open(cfg.model_dir + 'meter_{}_model_{}.pkl'.format(j, i), 'wb') as fout:
                pickle.dump(model, fout)
    return rmse

def evaluate(models, test_X:pd.DataFrame):
    print("------------- Predicting Test -------------")
    res = []
    step_size = 50000
    i = 0
    for _ in tqdm(range(int(np.ceil(test_X.shape[0] / step_size)))):
        res.append(np.expm1(sum([model.predict(test_X.iloc[i:i + step_size]) for model in models]) / cfg.folds))
        i += step_size
    res = np.concatenate(res)

    # %% submission. all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 100000 and the array at index 416 has size 97600
    sample_submission = pd.read_csv(cfg.data_dir + "/sample_submission.csv")
    sample_submission['meter_reading'] = res
    sample_submission.loc[sample_submission['meter_reading'] < 0, 'meter_reading'] = 0
    sample_submission.to_csv('submission.csv', index=False)


def load_models_per_meter():
    models_per_meter_id = []
    for meter_id in range(4):
        models = []
        for i in range(cfg.folds):
            with open(cfg.model_dir + 'meter_{}_model_{}.pkl'.format(meter_id, i), 'rb') as fout:
                model = pickle.load(fout)
                models.append(model)
        models_per_meter_id.append(models)
    return models_per_meter_id


def plotting_train_prediction(X,Y, use_important_features=False):
    print("------------- plotting train predictions-------------")
    if use_important_features:
        important_features_dataframes = get_important_features_dataframes()
        important_features_per_meter = [dataframe[dataframe['Value'] > 3]['Feature'] for dataframe in important_features_dataframes]
    models_per_meter = load_models_per_meter()
    for meter_id in X['meter'].value_counts(dropna=False).index.to_list():
        important_feat = important_features_per_meter[meter_id]
        models = models_per_meter[meter_id]
        print("--------- predicting train for meter {} ---------".format(meter_id))
        indices_by_meter_id = list(X[X['meter'] == meter_id].index)
        train_X_meter = X.loc[indices_by_meter_id]
        train_y_meter = Y.loc[indices_by_meter_id]
        train_X_meter.drop('meter', inplace=True, axis='columns')
        ground_y = train_y_meter.values
        train_X_meter = train_X_meter[important_feat]
        predictions = sum([model.predict(train_X_meter, num_iteration=model.best_iteration) for model in models]) / cfg.folds
        predictions[predictions < 0] = 0
        plt.figure()
        sns.distplot(predictions, label='pred')
        sns.distplot(ground_y, label='ground_truth')
        plt.title("meter {}".format(METER_DIC[meter_id]))
        print("RMSE score : \t {}".format(mean_squared_error(ground_y, predictions)))
        plt.legend()
        plt.show()
        plt.savefig(cfg.model_dir +"{}_dist_train.png".format(METER_DIC[meter_id]))
        if not use_important_features:
            # %%see which variables are the most relevant
            feature_imp = pd.DataFrame(sorted(zip(models[-1].feature_importance(), models[-1].feature_name()), reverse=True),
                                       columns=['Value', 'Feature'])
            feature_imp.to_pickle(path=cfg.feature_dir + "meter_{}.pkl".format(meter_id))
            plt.figure(figsize=(10, 5))
            sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
            plt.title('LightGBM FEATURES {}'.format(METER_DIC[meter_id]))
            plt.tight_layout()
            plt.show()

            print("")


def evaluate_per_meter(X_test, use_important_features=False):
    models_per_meter_id = []
    for meter_id in X_test['meter'].value_counts(dropna=False).index.to_list():
        models = []
        for i in range(cfg.folds):
            with open(cfg.model_dir + 'meter_{}_model_{}.pkl'.format(meter_id, i), 'rb') as fout:
                model = pickle.load(fout)
                models.append(model)

        models_per_meter_id.append(models)
    if use_important_features:
        important_features_dataframes = get_important_features_dataframes()
        important_features_per_meter = [dataframe[dataframe['Value'] > 3]['Feature'] for dataframe in important_features_dataframes]
    gc.collect()
    print("------------- Predicting Test -------------")
    res = pd.read_csv(cfg.data_dir + "/sample_submission.csv")
    res.set_index('row_id', drop=False, inplace=True)
    for meter_id in X_test['meter'].value_counts(dropna=False).index.to_list():
        important_feat = important_features_per_meter[meter_id]
        models = models_per_meter_id[meter_id]
        print("--------- predicting for meter {} ---------".format(meter_id))
        indices_by_meter_id = list(X_test[X_test['meter'] == meter_id].index)
        X_test_meter = X_test.loc[indices_by_meter_id]
        X_test_meter.drop(['meter'], inplace=True, axis='columns')
        X_test_meter = X_test_meter[important_feat]
        tmp = np.expm1(sum([model.predict(X_test_meter, num_iteration=model.best_iteration) for model in models]) / cfg.folds)
        res.loc[indices_by_meter_id, 'meter_reading'] = tmp
        gc.collect()
    res.loc[res['meter_reading'] < 0, 'meter_reading'] = 0
    res.to_csv('submission_aggregated_LGBM.csv', index=False)


if __name__ == '__main__':
    # if not os.path.isfile(cfg.results_df):
    #     results_df = pd.DataFrame(columns=['title', 'detail', 'dataset', 'model_name', 'rmse'])
    #     results_df.to_pickle(path=cfg.results_df)
    # else:
    #     results_df = pd.read_pickle(path=cfg.results_df)
    print("------------- Reading Train from path -------------")
    train_X_total = pd.read_pickle(path=cfg.ready_dir + 'train_X.pkl')
    train_Y_total = pd.read_pickle(path=cfg.ready_dir + 'train_y.pkl')
    test_X = pd.read_pickle(path=cfg.ready_dir + 'test_X.pkl')
    test_X = test_X.set_index('row_id', drop=True)
    train_X_total = train_X_total
    # best_params_for_each_meter = use_optuna_find_best_parameters(train_X_total, train_Y_total, sample=True, use_important_features=True)
    # with open(cfg.model_dir + 'params_per_meter.pkl', 'wb') as fout:
    #     pickle.dump(best_params_for_each_meter, fout)
    # with open(cfg.model_dir + 'params_per_meter.pkl', 'rb') as fread:
    #     best_params_for_each_meter = pickle.load(fread)
    # rmse = training_per_meter_with_optuna(train_X_total, train_Y_total, best_params_for_each_meter, use_important_features=True)
    # print("RMSE SCORE : {}".format(rmse))
    # plotting_train_prediction(train_X_total, train_Y_total, use_important_features=True)
    evaluate_per_meter(test_X, use_important_features=True)
    print("------------- Saving results in results csv -------------")
    # results = build_result_line("Experiment 5", "smart outlier removal + regression for missing values with features statistics + params with optuna", 'val', "ensemble lgb dividing to meter ", rmse)
    # results_df = results_df.append([results], ignore_index=True)
    # results_df.to_pickle(path=cfg.results_df)
    # results_df.to_csv(path_or_buf=cfg.results_csv, index=False)
    # with open(r'/home/dorliv/Desktop/KaggleFinal/results/features_important_meter_0.pkl', 'rb') as fread:
    #     feature_imp = pickle.load(fread)
    # feature_imp = feature_imp[feature_imp['Value'] > 15]
    # fig=plt.figure()
    # sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    # plt.title('LightGBM FEATURES {}'.format(METER_DIC[0]))
    # plt.tight_layout()
    # plt.show()
    # fig.savefig(cfg.eda_dir + 'features.png')
    # print("")
