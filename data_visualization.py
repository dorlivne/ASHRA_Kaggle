import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import CONFIG as cfg
FILES = ['test.csv', 'building_metadata.csv', 'train.csv', 'weather_test.csv', 'sample_submission.csv', 'weather_train.csv']

## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def extract_data(data_dir=cfg.data_dir):
    root_dir = data_dir
    train_df = pd.read_csv(root_dir + 'train.csv')
    weather_train_df = pd.read_csv(root_dir + 'weather_train.csv')
    test_df = pd.read_csv(root_dir + 'test.csv')
    weather_test_df = pd.read_csv(root_dir + 'weather_test.csv')
    building_meta_df = pd.read_csv(root_dir + 'building_metadata.csv')
    # train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    return train_df, weather_train_df, test_df, weather_test_df, building_meta_df

def load_data(data_dir=cfg.df_data_dir):
    root_dir = data_dir
    train_df = pd.read_pickle(root_dir + 'train_df.pkl')
    weather_train_df = pd.read_pickle(root_dir + 'weather_train_df.pkl')
    test_df = pd.read_pickle(root_dir + 'test_df.pkl')
    weather_test_df = pd.read_pickle(root_dir + 'weather_test_df.pkl')
    building_meta_df = pd.read_pickle(root_dir + 'building_metadata_df.pkl')
    # train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    return train_df, weather_train_df, test_df, weather_test_df, building_meta_df


def merge_building_data(train_df, test_df, building_meta_df):
    temp_df = train_df[['building_id']]
    temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')
    del temp_df['building_id']
    train_df = pd.concat([train_df, temp_df], axis=1)

    temp_df = test_df[['building_id']]
    temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')

    del temp_df['building_id']
    test_df = pd.concat([test_df, temp_df], axis=1)
    del temp_df, building_meta_df
    return train_df, test_df


def merge_weather_data(train_df, test_df, weather_train_df, weather_test_df):
    temp_df = train_df[['site_id', 'timestamp']]
    temp_df = temp_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

    del temp_df['site_id'], temp_df['timestamp']
    train_df = pd.concat([train_df, temp_df], axis=1)

    temp_df = test_df[['site_id', 'timestamp']]
    temp_df = temp_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')

    del temp_df['site_id'], temp_df['timestamp']
    test_df = pd.concat([test_df, temp_df], axis=1)

    del temp_df, weather_train_df, weather_test_df
    return train_df, test_df


if __name__ == '__main__':
    # train_df, weather_train_df, test_df, weather_test_df,\
    # building_meta_df = extract_data()
    # train_df = reduce_mem_usage(train_df)
    # weather_train_df = reduce_mem_usage(weather_train_df)
    # test_df = reduce_mem_usage(test_df)
    # weather_test_df = reduce_mem_usage(weather_test_df)
    # building_meta_df = reduce_mem_usage(building_meta_df)
    # train_df.to_pickle(path=cfg.df_data_dir + "train_df.pkl")
    # weather_train_df.to_pickle(path=cfg.df_data_dir + "weather_train_df.pkl")
    # test_df.to_pickle(path=cfg.df_data_dir + "test_df.pkl")
    # weather_test_df.to_pickle(path=cfg.df_data_dir + "weather_test_df.pkl")
    # building_meta_df.to_pickle(path=cfg.df_data_dir + "building_metadata_df.pkl")
    train_df, weather_train_df, test_df, weather_test_df,\
    building_meta_df = load_data()
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])
    weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])
    building_meta_df['primary_use'] = building_meta_df['primary_use'].astype('category')
    train_df, test_df = merge_building_data(train_df, test_df, building_meta_df)
    train_df, test_df = merge_weather_data(train_df, test_df, weather_train_df, weather_test_df)
    print(train_df.head())
