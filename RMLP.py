from keras.layers import *
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.losses import mean_squared_error as mse_loss
from data_manipulation import *
from keras import optimizers
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import numpy as np
import pandas as pd
import random
import gc
import pickle
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras import backend as K

K.set_learning_phase(1) #set learning phase

def train_model(keras_model:Model, X_t, y_train, batch_size, epochs, X_v, y_valid, fold, meter=None,
                patience=3, model_dir=cfg.MLP_dir, schedule=cfg.learning_rate_scheduler):
    early_stopping = EarlyStopping(patience=patience, verbose=1)
    if not meter:
        model_checkpoint = ModelCheckpoint(model_dir + "model_".format(meter) + str(fold) + ".hdf5",
                                           save_best_only=True, verbose=1, monitor='val_root_mean_squared_error',
                                           mode='min')
    else:
        model_checkpoint = ModelCheckpoint(model_dir + "meter_{}_model_".format(meter) + str(fold) + ".hdf5",
                                           save_best_only=True, verbose=1, monitor='val_root_mean_squared_error',
                                           mode='min')
    learning_rate_schedule = LearningRateScheduler(schedule=schedule)

    _ = keras_model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs,
                           validation_data=(X_v, y_valid), verbose=1,
                           callbacks=[learning_rate_schedule, early_stopping, model_checkpoint])
    if not meter:
        keras_model = load_model(model_dir + "model_".format(meter) + str(fold) + ".hdf5",
                                 custom_objects={'root_mean_squared_error': root_mean_squared_error})
    else:
        keras_model = load_model(model_dir + "meter_{}_model_".format(meter) + str(fold) + ".hdf5",
                                 custom_objects={'root_mean_squared_error': root_mean_squared_error})
    return keras_model


def get_keras_data(df):
    X = {col: np.array(df[col]) for col in df.columns}
    return X

def get_keras_data_RMLP(df, feature_names):
    X = {feat: df[:, :, col_id] for col_id, feat in zip(range(df.shape[-1]), feature_names)}
    return X

def preprocess_data_for_RMLP(X_train:pd.DataFrame, RMLP=False, test_signal=False):
    if not RMLP:
        categoricals = ["site_id", "building_id", "primary_use", "hour", "weekday", "meter", "month", "dayofyear"]

    elif test_signal:
        categoricals = ["row_id", "site_id", "building_id", "primary_use", "hour", "weekday", "meter", "month", "dayofyear", "year"]

    else:
        categoricals = ["site_id", "building_id", "primary_use", "hour", "weekday", "meter", "month",
                        "dayofyear", "year"]

    numericals = ["square_feet", "age", "air_temperature", "cloud_coverage",
                  "dew_temperature", 'building_min', 'building_max', 'sea_level_pressure']

    feat_cols = categoricals + numericals
    X = X_train[feat_cols]
    if test_signal:
        X['row_id'] += 1  # bias for test row- to solve padded seq problem
    return X


def transform_to_3_dim(X):
    columns_shape = X.shape[1]
    X['single_input'] = X.apply(tuple, axis=1).apply(list)  # organizes the series in a list
    X['single_input'] = X['single_input'].apply(lambda x: [list(x)]) # organizes the list in a list, for concatenation effect when using cumsum
    X['hourly_samples'] = X['single_input'].cumsum()  # concatenate the series
    ans = np.asarray(X.hourly_samples.values[-1])  # we take the last sample which holds all the samples concatenated
    if np.shape(ans)[0] != 24:
        ans = np.reshape(ans, (1, ans.shape[0], ans.shape[-1]))
        ans = pad_sequences(ans, 24, dtype='float32').tolist()
    ans = np.reshape(ans, (-1, 24, columns_shape))
    return ans


def process_batch(X:pd.DataFrame, Y=None):
    X_time_series_batch, Y_batch = [], []
    if Y is not None:
        X['meter_reading'] = Y.values
    feat_columns = X.shape[1] - 1  # -1 is because of year
    for year in list(X['year'].unique()):
        print("Adjusting for year {}".format(year))
        for day_in_year in tqdm(list(X['dayofyear'].unique())):
            criteria = (X['dayofyear'] == day_in_year) & (X['year'] == year)
            X_to_criterion = X.loc[criteria]
            X_to_criterion.drop('year', inplace=True, axis='columns')
            sorted_by_hour = X_to_criterion.groupby(['building_id', 'meter']).\
                apply(lambda x: transform_to_3_dim(x.sort_values(["hour"])))
            sorted_by_hour = sorted_by_hour.values.tolist()
            padded_sequences_to_whole_day = np.reshape(sorted_by_hour, (-1, 24, feat_columns))

            if Y is not None:
                X_tmp = padded_sequences_to_whole_day[:, :, :-1]  # features
                y_tmp = padded_sequences_to_whole_day[:, :, -1]  # meter_Reading
                Y_batch.append(y_tmp)
            else:
                X_tmp = padded_sequences_to_whole_day  # features
            X_time_series_batch.append(X_tmp)
    if Y is not None:
        return X_time_series_batch, Y_batch
    else:
        return X_time_series_batch

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))

# def get_batch_from_dataframe(X_train:pd.DataFrame, Y_train:pd.DataFrame):
#     min_dayyear = X_train['dayofyear'].min()  # usually 0
#     max_dayyear = X_train['dayofyear'].max()  # usually 364
#     day_of_year_batch_values = list(random.sample(range(min_dayyear, max_dayyear), cfg.batch_size))
#     # take only the samples that correspond to the dayofyear we sampled
#     batch_criteria = np.isin(X_train['dayofyear'].values, day_of_year_batch_values)
#     X_batch = X_train[batch_criteria]
#     indices = list(X_batch.index)
#     Y_batch = Y_train.loc[indices]
#     X_batch_timeseries_by_hour, Y_batch_timeseries_by_hour = process_batch(X_batch, Y_batch) # get batch of samples and organize them for LSTM
#     return X_batch_timeseries_by_hour, Y_batch_timeseries_by_hour

def load_RMLP():
    print("------------- Loading RMLP models -------------")
    models = []
    for i in range(cfg.MLP_folds):
        keras_model = load_model(cfg.RMLP_dir + "model_" + str(i) + ".hdf5",
                                 custom_objects={'root_mean_squared_error': root_mean_squared_error})
        models.append(keras_model)
    return models

def RMLP(features, drop_rate=0.2, metrics=root_mean_squared_error, loss='mse', lr=0.001):
    # # Inputs
    # # cat = ['site_id','building_id','meter','primary_use','hour','weekday','dayofyear','month']
    # inputs_dic_index = {}
    # inputs = []
    # for i, feat in enumerate(features):
    #     shape = [24]
    #     tmp_in = Input(shape=shape, name=feat)
    #     inputs.append(tmp_in)
    #     inputs_dic_index[feat] = i
    # # Embeddings layers
    # emb_site_id = Embedding(16, 2, input_length=24)(inputs[inputs_dic_index['site_id']])
    # # 15 site's multiplied by 6 of primary use (there is 5 primary use
    # # that form most of the samples : https://www.kaggle.com/angelapuc/model-lgbm-of-final-ds-notebook)
    # emb_building_id = Embedding(1449, 90, input_length=24)(inputs[inputs_dic_index['building_id']])
    # emb_meter = Embedding(4, 3, input_length=24)(inputs[inputs_dic_index['meter']]) # steam, hot water versus electricity versus cool water
    # emb_primary_use = Embedding(16, 6, input_length=24)(inputs[inputs_dic_index['primary_use']])  # see link above
    # emb_hour = Embedding(24, 3, input_length=24)(inputs[inputs_dic_index['hour']])  # morning, noon and evening
    # emb_weekday = Embedding(7, 2, input_length=24)(inputs[inputs_dic_index['weekday']]) # weekend and not weekend
    # emb_yearday = Embedding(365, 2, input_length=24)(inputs[inputs_dic_index['dayofyear']])  # holidya
    # emb_month = Embedding(12, 4, input_length=24)(inputs[inputs_dic_index['month']])
    # expand = Lambda(lambda x: tf.expand_dims(x, axis=-1))
    # main_l = concatenate([
    #        emb_site_id
    #      , emb_building_id
    #      , emb_meter
    #      , emb_primary_use
    #      , emb_hour
    #      , emb_weekday
    #      , emb_month
    #      , emb_yearday
    #      , expand(inputs[inputs_dic_index['square_feet']])
    #      , expand(inputs[inputs_dic_index['age']])
    #      , expand(inputs[inputs_dic_index['air_temperature']])
    #      , expand(inputs[inputs_dic_index['cloud_coverage']])
    #      , expand(inputs[inputs_dic_index['dew_temperature']])
    #      , expand(inputs[inputs_dic_index['building_min']])
    #      , expand(inputs[inputs_dic_index['building_max']])
    #      , expand(inputs[inputs_dic_index['sea_level_pressure']])
    # ])
    # lstm_layer = LSTM(cfg.lstm_units, return_sequences=True, input_shape=(None, 120))(main_l) # calculation of the rightmost shape of main_l
    # lstm_layer_post_batch = TimeDistributed(BatchNormalization(), input_shape=(24, cfg.lstm_units))(lstm_layer)
    # dense_1 = TimeDistributed(Dense(32, activation='relu'))(lstm_layer_post_batch)
    # dense_1_post_drop = TimeDistributed(Dropout(drop_rate))(dense_1)
    # dense_1_post_drop = TimeDistributed(BatchNormalization())(dense_1_post_drop)
    # prediction_seq = TimeDistributed(Dense(1))(dense_1_post_drop)
    # model = Model(inputs, prediction_seq)
    # model.compile(optimizer=Adam(lr=lr),
    #               loss=mse_loss,
    #               metrics=[root_mean_squared_error])
    model = Sequential()
    # model.add(LSTM(cfg.lstm_units, return_sequences=True, input_shape=(None, features.shape[-1])))
    model.add(LSTM(cfg.lstm_units, return_sequences=True, input_shape=(None, features.shape[-1])))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=[metrics])
    return model


def evaluate_RMLP(X_test_df, first=False):
    if first:
        print("------------ Creating Time Series data ------------")
        X_test = process_batch(X_test_df)
        with open(cfg.ready_dir + 'X_test_timeseries.pkl', 'wb') as fout:
            pickle.dump(X_test, fout)
    else:
        print("------------ Loading Time Series data ------------")
        with open(cfg.ready_dir + 'X_test_timeseries.pkl', 'rb') as fout:
            X_test = pickle.load(fout)
    X_test = np.vstack(X_test)
    X_test_feat = X_test[:, :, 1:]
    row_col = X_test[:, :, 0].astype(np.int32)  # row id is the first feature
    max_row_id_df = X_test_df.row_id.max()
    max_row_id_timeseries = np.max(row_col)
    assert max_row_id_df == max_row_id_timeseries, "max row of dataframe is not the same as timeseries version"
    print("------------- Predicting Test -------------")
    models = load_RMLP()
    res = pd.read_csv(cfg.data_dir + "/sample_submission.csv")
    res.set_index('row_id', drop=False, inplace=True)
    i = 0
    step_size = 50000
    for _ in tqdm(range(int(np.ceil(X_test_feat.shape[0] / step_size)))):
        for_prediction = X_test_feat[i:min(i + step_size, X_test_feat.shape[0]), :, :]
        indices_for_res = row_col[i:min(i + step_size, X_test_feat.shape[0]), :] - 1  # correct the bias
        tmp = np.expm1(sum([model.predict(for_prediction) for model in models]) / cfg.MLP_folds)
        tmp = tmp[indices_for_res >= 0, :]  # take only the valid rows
        indices_for_res = indices_for_res[indices_for_res >= 0]
        res.loc[indices_for_res, 'meter_reading'] = np.squeeze(tmp)
        i += step_size
    res.loc[res['meter_reading'] < 0, 'meter_reading'] = 0
    res.to_csv('submission_aggregated_RMLP.csv', index=False)


def plotting_train_RMLP_prediction(X_df, Y_df):
    model_name = "RMLP"
    models = load_RMLP()
    print("------------ Loading Time Series data ------------")
    with open(cfg.ready_dir + 'X_timeseries.pkl', 'rb') as fout:
        X = pickle.load(fout)
    with open(cfg.ready_dir + 'Y_timeseries.pkl', 'rb') as fout:
        Y = pickle.load(fout)
    X = np.vstack(X)
    Y = np.expand_dims(np.vstack(Y), axis=-1)
    X = X[:100000, :, :]
    Y = Y[:100000, :]
    print("--------- predicting train---------\n")
    ground_y = Y.ravel()  # create flattened array
    i = 0
    res = np.zeros_like(Y)
    step_size = 50000
    for _ in tqdm(range(int(np.ceil(X.shape[0] / step_size)))):
        for_prediction = X[i:min(i + step_size, X.shape[0]), :, :]
        res[i:min(i + step_size, X.shape[0]), :, :] = sum([model.predict(for_prediction) for model in models]) / cfg.MLP_folds
        i += step_size
    res = res.ravel()
    print("RMSE score : \t {}".format(mean_squared_error(ground_y, res)))
    res[res < 0] = 0
    fig, axes = plt.subplots(1, 1)
    sns.distplot(res, label='pred', ax=axes)
    sns.distplot(ground_y, label='ground_truth', ax=axes)
    axes.set_title("pred_Vs_GT_{}".format(model_name))
    axes.legend()
    fig.show()
    fig.savefig("{}_dist_train.png".format(model_name))


def train_models_using_RMLP(X_df, Y_df, first=False):
    epochs = 250
    models = []
    if first:
        print("------------ Creating Time Series data ------------")
        X, Y = process_batch(X_df.copy(), Y_df.copy())
        with open(cfg.ready_dir + 'X_timeseries.pkl', 'wb') as fout:
            pickle.dump(X, fout)
        with open(cfg.ready_dir + 'Y_timeseries.pkl', 'wb') as fout:
            pickle.dump(Y, fout)
    else:
        print("------------ Loading Time Series data ------------")
        with open(cfg.ready_dir + 'X_timeseries.pkl', 'rb') as fout:
            X = pickle.load(fout)
        with open(cfg.ready_dir + 'Y_timeseries.pkl', 'rb') as fout:
            Y = pickle.load(fout)
    X = np.vstack(X)
    Y = np.expand_dims(np.vstack(Y), axis=-1)
    X_df.drop('year', axis='columns', inplace=True)
    for fold_n in range(cfg.MLP_folds):
        print('RMLP:', fold_n)
        X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=1 / cfg.MLP_folds)
        X_train = get_keras_data_RMLP(X_train, X_df.columns) # TODO delete if we return to prior version without embedding
        X_valid = get_keras_data_RMLP(X_valid, X_df.columns)
        keras_rmlp = RMLP(X_df.columns)
        keras_rmlp.summary()
        trained_rmlp = train_model(keras_model=keras_rmlp, X_t=X_train, y_train=y_train,
                                   batch_size=cfg.batch_size, epochs=epochs, patience=20,
                                   X_v=X_valid, y_valid=y_valid, fold=fold_n, model_dir=cfg.RMLP_dir,
                                   schedule=cfg.learning_rate_scheduler_RMLP)
        models.append(trained_rmlp)
        print('*' * 50)


if __name__ == '__main__':
    train_X_total = pd.read_pickle(path=cfg.ready_dir + 'train_X.pkl')
    train_Y_total = pd.read_pickle(path=cfg.ready_dir + 'train_y.pkl')
    train_X_total = preprocess_data_for_RMLP(X_train=train_X_total, RMLP=True)
    train_models_using_RMLP(train_X_total, train_Y_total, first=False)
    # plotting_train_RMLP_prediction(train_X_total, train_Y_total)
    test_X = pd.read_pickle(path=cfg.ready_dir + 'test_X.pkl')
    test_X = preprocess_data_for_RMLP(test_X, RMLP=True, test_signal=True)
    evaluate_RMLP(test_X, first=False)