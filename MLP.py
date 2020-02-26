from keras.layers import Input, Dropout, Dense, Embedding, concatenate, BatchNormalization, Flatten
from sklearn.model_selection import StratifiedKFold
from keras.models import Model, load_model
from keras.losses import mean_squared_error as mse_loss
from data_manipulation import *
from keras.optimizers import  Adam
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from RMLP import train_model, root_mean_squared_error, get_keras_data, preprocess_data_for_RMLP


def MLP(feature_names, dropout1, dropout2, dropout3, dropout4, dense_dim_1, dense_dim_2, dense_dim_3, dense_dim_4, lr):
    # Inputs
    inputs_dic_index = {}
    inputs = []
    for i, feat in enumerate(feature_names):
        tmp_in = Input(shape=[1], name=feat)
        inputs.append(tmp_in)
        inputs_dic_index[feat] = i
    # Embeddings layers
    emb_site_id = Embedding(16, 2)(inputs[inputs_dic_index['site_id']])
    emb_building_id = Embedding(1449, 6)(inputs[inputs_dic_index['building_id']])
    emb_primary_use = Embedding(16, 2)(inputs[inputs_dic_index['primary_use']])
    emb_hour = Embedding(24, 3)(inputs[inputs_dic_index['hour']])
    emb_weekday = Embedding(7, 2)(inputs[inputs_dic_index['weekday']])
    emb_yearday = Embedding(365, 2)(inputs[inputs_dic_index['dayofyear']])
    emb_month = Embedding(12, 4)(inputs[inputs_dic_index['month']])
    concat_emb = concatenate([
           Flatten()(emb_site_id)
         , Flatten()(emb_building_id)
         , Flatten()(emb_primary_use)
         , Flatten()(emb_hour)
         , Flatten()(emb_weekday)
         , Flatten()(emb_month)
         , Flatten()(emb_yearday)
    ])
    categ = Dropout(dropout1)(Dense(dense_dim_1, activation='relu')(concat_emb))
    categ = BatchNormalization()(categ)
    categ = Dropout(dropout2)(Dense(dense_dim_2, activation='relu')(categ))

    # main layer
    main_l = concatenate([
        categ
        , inputs[inputs_dic_index['square_feet']]
        , inputs[inputs_dic_index['age']]
        , inputs[inputs_dic_index['air_temperature']]
        , inputs[inputs_dic_index['cloud_coverage']]
        , inputs[inputs_dic_index['dew_temperature']]
        # , inputs[inputs_dic_index['precip_depth_1_hr']]
    ])

    main_l = Dropout(dropout3)(Dense(dense_dim_3, activation='relu')(main_l))
    main_l = BatchNormalization()(main_l)
    main_l = Dropout(dropout4)(Dense(dense_dim_4, activation='relu')(main_l))

    # output
    output = Dense(1)(main_l)

    model = Model(inputs, output)
    # model.summary()
    model.compile(optimizer=Adam(lr=lr),
                  loss=mse_loss,
                  metrics=[root_mean_squared_error])
    return model


def train_models_using_MLP(X, Y):
    # batch_size = 1024
    epochs = 10
    models = []
    seed = 666
    kf = StratifiedKFold(n_splits=cfg.MLP_folds, shuffle=True, random_state=seed)
    for meter_id in X['meter'].value_counts(dropna=False).index.to_list():
        indices_by_meter_id = list(X[X['meter'] == meter_id].index)
        train_X_meter = X.loc[indices_by_meter_id]
        train_y_meter = Y.loc[indices_by_meter_id]
        train_X_meter.drop('meter', inplace=True, axis='columns')
        for fold_n, (train_index, valid_index) in enumerate(kf.split(train_X_meter, train_X_meter['building_id'])):
            print('MLP:', fold_n)
            X_train, X_valid = train_X_meter.iloc[train_index], train_X_meter.iloc[valid_index]
            y_train, y_valid = train_y_meter.iloc[train_index], train_y_meter.iloc[valid_index]
            X_t = get_keras_data(X_train)
            X_v = get_keras_data(X_valid)
            keras_model = MLP(feature_names=train_X_meter.columns, dense_dim_1=64,
                              dense_dim_2=32,
                              dense_dim_3=32,
                              dense_dim_4=16,
                              dropout1=0.2, dropout2=0.1, dropout3=0.1, dropout4=0.1, lr=0.001)
            mod = train_model(keras_model, X_t, y_train, cfg.batch_size, epochs, X_v, y_valid, fold_n, meter=meter_id,
                              patience=3)
            models.append(mod)
            print('*' * 50)


def load_models(model_dir, model_name):
    print("------------- Loading {} models -------------".format(model_name))
    models_per_meter_id = []
    for j in range(4):
        models = []
        for i in range(cfg.MLP_folds):
            keras_model = load_model(model_dir + "meter_{}_model_".format(j) + str(i) + ".hdf5",
                                     custom_objects={'root_mean_squared_error': root_mean_squared_error})
            models.append(keras_model)
        models_per_meter_id.append(models)
    return models_per_meter_id


def evaluate_MLP(X_test, model_dir=cfg.MLP_dir):
    model_name = "MLP"
    models_per_meter_id = load_models(model_dir, model_name)
    print("------------- Predicting Test -------------")
    res = pd.read_csv(cfg.data_dir + "/sample_submission.csv")
    res.set_index('row_id', drop=False, inplace=True)
    for meter_id in X_test['meter'].value_counts(dropna=False).index.to_list():
        print("--------- predicting for meter {} ---------".format(meter_id))
        models = models_per_meter_id[meter_id]
        indices_by_meter_id = list(X_test[X_test['meter'] == meter_id].index)
        X_test_meter = X_test.loc[indices_by_meter_id]
        X_test_meter.drop('meter', inplace=True, axis='columns')
        i = 0
        res_tmp = np.zeros((X_test_meter.shape[0]), dtype=np.float32)
        step_size = 50000
        for _ in tqdm(range(int(np.ceil(X_test_meter.shape[0] / step_size)))):
            for_prediction = get_keras_data(X_test_meter.iloc[i:i + step_size])
            res_tmp[i:min(i + step_size, X_test_meter.shape[0])] = np.expm1(sum([model.predict(for_prediction)[:, 0] for model in models]) / cfg.MLP_folds)
            i += step_size
        # tmp = np.expm1(sum([model.predict(get_keras_data(X_test_meter)) for model in models]) / cfg.MLP_folds)
        res.loc[indices_by_meter_id, 'meter_reading'] = res_tmp
        gc.collect()
    res.loc[res['meter_reading'] < 0, 'meter_reading'] = 0
    res.to_csv('submission_aggregated_{}.csv'.format(model_name), index=False)


def plotting_train_prediction(X, Y, model_dir=cfg.MLP_dir):
    model_name = "MLP"
    models_per_meter_id = load_models(model_dir, model_name)
    for meter_id in X['meter'].value_counts(dropna=False).index.to_list():
        print("--------- predicting train for meter {} ---------".format(meter_id))
        indices_by_meter_id = list(X[X['meter'] == meter_id].index)
        train_X_meter = X.loc[indices_by_meter_id]
        train_y_meter = Y.loc[indices_by_meter_id]
        models = models_per_meter_id[meter_id]
        train_X_meter.drop('meter', inplace=True, axis='columns')
        ground_y = train_y_meter.values
        i = 0
        res = np.zeros((train_X_meter.shape[0]), dtype=np.float32)
        step_size = 50000
        for _ in tqdm(range(int(np.ceil(train_X_meter.shape[0] / step_size)))):
            for_prediction = get_keras_data(train_X_meter.iloc[i:i + step_size])
            res[i:min(i + step_size, train_X_meter.shape[0])] = sum([model.predict(for_prediction)[:, 0] for model in models]) / cfg.MLP_folds
            i += step_size
        print("RMSE score : \t {}".format(mean_squared_error(ground_y, res)))
        res[res < 0] = 0
        fig, axes = plt.subplots(1, 1, figsize=(14, 20), dpi=100)
        sns.distplot(res, label='pred', ax=axes)
        sns.distplot(ground_y, label='ground_truth', ax=axes)
        axes.set_title("meter_{}_pred_Vs_GT_{}".format(meter_id, model_name))
        axes.legend()
        fig.show()
        fig.savefig("meter_{}_{}_dist_train.png".format(meter_id, model_name))


if __name__ == '__main__':
    train_X_total = pd.read_pickle(path=cfg.ready_dir + 'train_X.pkl')
    train_Y_total = pd.read_pickle(path=cfg.ready_dir + 'train_y.pkl')
    train_X_total = preprocess_data_for_RMLP(X_train=train_X_total, RMLP=False)
    train_models_using_MLP(train_X_total, train_Y_total)
    test_X = pd.read_pickle(path=cfg.ready_dir + 'test_X.pkl')
    test_X = test_X.set_index('row_id', drop=True)
    test_X = preprocess_data_for_RMLP(test_X, RMLP=False, test_signal=True)
    plotting_train_prediction(train_X_total, train_Y_total)
    evaluate_MLP(test_X)