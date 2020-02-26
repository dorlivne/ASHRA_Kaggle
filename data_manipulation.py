from data_visualization import *
from sklearn.preprocessing import LabelEncoder
from utils.imputer import nan_imputer


def make_is_bad_zero(Xy_subset, min_interval=24, summer_start=3000, summer_end=7500):
    """Helper routine for 'find_bad_zeros'.

    This operates upon a single dataframe produced by 'groupby'. We expect an
    additional column 'meter_id' which is a duplicate of 'meter' because groupby
    eliminates the original one."""
    meter = Xy_subset.meter_id.iloc[0]
    is_zero = Xy_subset.meter_reading == 0
    if meter == 0:  # electricity
        # Electrical meters should never be zero. Keep all zero-readings in this table so that
        # they will all be dropped in the train set.
        return is_zero
    # the meter reading changes from zero to not zero from the i obs to the i+1 obs
    transitions = (is_zero != is_zero.shift(1))
    all_sequence_ids = transitions.cumsum()  # cumulative sum, we now know whats the size of each sequence
    ids = all_sequence_ids[is_zero].rename("ids")  # we care only about a sequence which is zero in the meter readings
    if meter in [2, 3]:
        # It's normal for steam and hotwater to be turned off during the summer
        not_summer = set(ids[(Xy_subset.timestamp_h < summer_start) |
                       (Xy_subset.timestamp_h > summer_end)].unique())
        is_bad = ids.isin(not_summer) & (ids.map(ids.value_counts()) >= min_interval)
        # each id with sequence bigger than 48 is bad news, thus we delete it
    elif meter == 1:  # chilledwater
        time_ids = ids.to_frame().join(Xy_subset.timestamp_h).set_index("timestamp_h").ids
        is_bad = ids.map(ids.value_counts()) >= min_interval

        # Cold water may be turned off during the winter
        jan_id = time_ids.get(0, False)
        dec_id = time_ids.get(8283, False)
        if (jan_id and dec_id and jan_id == time_ids.get(500, False) and
                dec_id == time_ids.get(8783, False)):
            is_bad = is_bad & (~(ids.isin(set([jan_id, dec_id]))))
    else:
        raise Exception(f"Unexpected meter type: {meter}")

    result = is_zero.copy()
    result.update(is_bad)
    return result

def find_bad_sitezero(X):
    """Returns indices of bad rows from the early days of Site 0 (UCF)."""
    return X[(X['timestamp'] < "2016-05-21 00:00:00") & (X.site_id == 0) & (X.meter == 0)].index

def find_bad_building1099(X, y):
    """Returns indices of bad rows (with absurdly high readings) from building 1099."""
    return X[(X.building_id == 1099) & (X.meter == 2) & (y > 3e4)].index

def find_bad_zeros(X, y):
    """Returns an Index object containing only the rows which should be deleted."""
    Xy = X.assign(meter_reading=y, meter_id=X.meter)
    is_bad_zero = Xy.groupby(["building_id", "meter"]).apply(make_is_bad_zero)
    return is_bad_zero[is_bad_zero].index.droplevel([0, 1])


def find_bad_rows(X:pd.DataFrame):
    y = X['meter_reading']
    return find_bad_zeros(X, y).union(find_bad_sitezero(X)).union(find_bad_building1099(X, y))


def print_missing_in_df(df: pd.DataFrame):
    for col in df.columns:
            missing_data = len(df) - df[col].count()
            if (missing_data > 0 or missing_data == 'NaN'):
                precentage_missing = round(100 * (missing_data / len(df)), 3)
                print(col, ':', missing_data, 'missing values is', str(precentage_missing), '% of total')


def interpolating_missing_values(df:pd.DataFrame):
    cols_fill_with_imputer = ['air_temperature', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_speed']
    cols_to_fill = df.columns
    cols_that_has_long_missing_seq = []
    nans_total = df[cols_to_fill].isnull().sum().sum()
    nans_filled = 0
    for col in cols_to_fill:
        has_long_missing_seq_flag = False

        nans_before = df[col].isnull().sum()
        dtype = np.float32
        if col in cfg.int_cols or col in ['timestamp']:
            # cols_that_has_long_missing_seq.append(col)
            continue
        if col in cols_fill_with_imputer:
            thd = 3
        else:
            # random high number, we dont want to use imputer for lag variables, we want to use those features to better estimate the core features
            thd = 100000
        print(f'filling short NaN series in col "{col}"')
        for sid in sorted(df.site_id.unique()):
            s = df[df['site_id'] == sid][col]
            idx_lt, idx_gte = get_nan_sequences(s, thd)  # higher then thd we predict with imputer
            interpolation = s.interpolate()
            if len(idx_lt) > 0:
                nan_standalone_indices = list(df[df['site_id'] == sid].iloc[idx_lt].index)
                values_interpolated = interpolation.iloc[idx_lt].values.astype(dtype)
                df.at[nan_standalone_indices, col] = values_interpolated
            if len(idx_gte) > 0:
                has_long_missing_seq_flag = True
        if not col in cols_fill_with_imputer:
            df[col].fillna(0, inplace=True)  # for the edge variables of lag ( only 16 samples)
        if has_long_missing_seq_flag:
            cols_that_has_long_missing_seq.append(col)
        nans_after = df[col].isnull().sum()
        print(f'\t\tnans before: {nans_before}')
        print(f'\t\tnans  after: {nans_after}')
        nans_filled += (nans_before - nans_after)
    print(f'Nans filled: {nans_filled}/{nans_total}:   {np.round(nans_filled / nans_total * 100, 2)}%')
    return df, cols_that_has_long_missing_seq


def filling_long_missing_seq(df: pd.DataFrame, cols_that_has_long_missing_seq):
    for tcol in cols_that_has_long_missing_seq:
            df = nan_imputer(data=df, tcol=tcol, window=24)
    return df


def merge_weather_plus_index_them_according_to_site_and_timestap(weather_train:pd.DataFrame, weather_test:pd.DataFrame):
    print("some features in weather data frame are totally missing from site id, so we put zero in them and gave a bias of 1 to all features")
    weather_train = add_rolling_statistics(weather_train)
    weather_test = add_rolling_statistics(weather_test)
    print(" ----------------- Adding rolling statistics to weather df-----------------")
    df = pd.concat(
        [
            weather_train,
            weather_test
        ],
        ignore_index=True, axis='rows'
    )
    unique_site_ids = sorted(np.unique(df['site_id']))
    df = df.set_index(['site_id', 'timestamp'], drop=False).sort_index()

    # construct full index w/o missing dates
    full_index = pd.MultiIndex.from_product(
        [
            unique_site_ids,
            pd.date_range(start='2015-12-31 19:00:00', end='2018-12-31 23:00:00', freq='H')
        ]
    )

    print(f'init shape: {df.shape}')
    df = df.reindex(full_index)
    print(f'full shape: {df.shape}')

    df['site_id'] = df.index.get_level_values(0).astype(np.uint8)
    df['timestamp'] = df.index.get_level_values(1)
    dtime_col = 'timestamp'
    df['hour'] = df[dtime_col].dt.hour.astype(np.uint8)
    df['weekday'] = df[dtime_col].dt.weekday.astype(np.uint8)
    df['weekofyear'] = df[dtime_col].dt.weekofyear.astype(np.uint8)
    df['dayofyear'] = df[dtime_col].dt.dayofyear.astype(np.uint16) - 1
    df['month'] = df['timestamp'].dt.month.astype(np.uint8)
    df['year'] = df[dtime_col].dt.year.astype(np.uint16)
    df['season'] = df.month.apply(convert_month_to_season)

    return df


def fill_missing_data(weather_train: pd.DataFrame, weather_test: pd.DataFrame):
    # weather_train['DataType'], weather_test['DataType'] = 'train', 'test'
    df = merge_weather_plus_index_them_according_to_site_and_timestap(weather_train, weather_test)
    df, cols_that_has_long_missing_seq = interpolating_missing_values(df)
    df = filling_long_missing_seq(df, cols_that_has_long_missing_seq)
    #     print("filling missing values for year_built feature according to mean by site")
    if cfg.verbose:
        print("filling missing values for cloud coverage feature according to mean by site and by hour and month")

    df['cloud_coverage'] = df['cloud_coverage'].fillna(round(df.groupby(['site_id', 'dayofyear', 'hour'])['cloud_coverage'].transform('mean'), 0))
    df['cloud_coverage'] = df['cloud_coverage'].fillna(round(df.groupby(['site_id', 'dayofyear'])['cloud_coverage'].transform('mean'), 0))
    if cfg.verbose:
        print("filling missing values for wind_direction feature according to mean by site and month")
    df['wind_direction'] = df['wind_direction'].fillna(round(
        df.groupby(['site_id', 'dayofyear', 'hour'])['wind_direction'].transform('mean'), 0))
    df['wind_direction'] = df['wind_direction'].fillna(round(
        df.groupby(['site_id', 'dayofyear'])['wind_direction'].transform('mean'), 0))
    df = df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
    print_missing_in_df(df)
    df.reset_index(inplace=True, drop=True)
    return df

def convert_month_to_season(month):
    if (month <= 2) | (month == 12):
        return 0  # winter
    elif month <= 5:
        return 1  # spring
    elif month <= 8:
        return 2  # summer
    elif month <= 11:
        return 3  # fall

def label_encoder(df, categorical_columns=None):
    """Encode categorical values as integers (0,1,2,3...) with pandas.factorize. """
    # if categorical_colunms are not given than treat object as categorical features
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col])
    return df

def feature_extraction(df:pd.DataFrame):
    primary_use_list = building_meta_df['primary_use'].unique()
    primary_use_dict = {key: value for value, key in enumerate(primary_use_list)}
    print('primary_use_dict: ', primary_use_dict)
    building_meta_df['primary_use'] = building_meta_df['primary_use'].map(primary_use_dict)
    print("skewing square feet variable with log(p)")
    df['square_feet'] = np.log1p(df['square_feet'])
    print("adding age feature")
    df['age'] = df['year_built'].max() - df['year_built'] + 1
    print("removing year_built bias")
    df['year_built'] = df['year_built'] - 1900
    return df


def remove_outliers(df: pd.DataFrame):
    df['timestamp_h'] = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    print("removing discrepancy in meter reading for site id 10 before May 2016")
    print("removing anomality values of building 1099")
    print("removing zero electricity meter reading ")
    print("removing sequences of +48 hours of zero reading of steam and hotwater except during the summer")
    print("removing sequences of +48 hours of zero reading of chilled water except during the winter")
    # criteria = (df['site_id'] == 0) & (df['timestamp'] < "2016-05-21 00:00:00")
    idx_to_drop = find_bad_rows(df)
    df.drop(idx_to_drop, axis='rows', inplace=True)
    df.drop(['timestamp_h'], inplace=True, axis='columns')
    return df

def get_nan_sequences(series: pd.Series, thld_nan: int = 2):
    """
    Given sequence with missing data, builds joint index
    from consecutive NaN blocks
    1) of len  < thld_nan
    2) of len >= thld_nan
    and returns them as 1-D np.arrays

    thld_nan >= 2
    solution is based on:
    https://stackoverflow.com/questions/42078259/indexing-a-numpy-array-using-a-numpy-array-of-slices
    """
    b = series.values
    idx0 = np.flatnonzero(np.r_[True, np.diff(np.isnan(b)) != 0, True])  # indices of nan variables
    count = np.diff(idx0)  # length of seq
    idx = idx0[:-1]
    # >=
    valid_mask_gte = (count >= thld_nan) & np.isnan(b[idx])
    out_idx = idx[valid_mask_gte]
    out_count = count[valid_mask_gte]

    if len(out_idx) == 0:
        out_seq = np.empty(shape=0)  # sequence of nan's
    else:
        out_seq = np.hstack([
            np.array(range(series, series + n))
            for (series, n) in zip(out_idx, out_count) # sequence of nan's
        ])
    # <
    valid_mask_standalone = (count < thld_nan) & np.isnan(b[idx])  # stand alone nan's
    out_idx = idx[valid_mask_standalone]
    out_count = count[valid_mask_standalone]

    if len(out_idx) == 0:
        out_standalone = np.empty(shape=0) #stand alone nan's
    else:
        out_standalone = np.hstack([
            np.array(range(st, st + n))
            for (st, n) in zip(out_idx, out_count)# stand alone nan's
        ])
    # check if out_seq + out_standalone = all NaNs
    assert len(out_seq) + len(out_standalone) == series.isnull().sum(), 'incorrect calculations'

    return out_standalone, out_seq


def drop_features(df:pd.DataFrame):
    drop_columns = ['timestamp', 'floor_count']
    print("features dropped: \t {}".format(drop_columns))
    df.drop(drop_columns, inplace=True, axis='columns')
    return df


def add_target_feature(train_dataframe:pd.DataFrame, test_dataframe: pd.DataFrame):
    df_group = train_dataframe.groupby('building_id')['meter_reading']
    building_mean = df_group.mean().astype(np.float16)
    building_median = df_group.median().astype(np.float16)
    building_min = df_group.min().astype(np.float16)
    building_max = df_group.max().astype(np.float16)
    building_std = df_group.std().astype(np.float16)

    train_dataframe['building_mean'] = train_dataframe['building_id'].map(building_mean)
    train_dataframe['building_median'] = train_dataframe['building_id'].map(building_median)
    train_dataframe['building_min'] = train_dataframe['building_id'].map(building_min)
    train_dataframe['building_max'] = train_dataframe['building_id'].map(building_max)
    train_dataframe['building_std'] = train_dataframe['building_id'].map(building_std)

    test_dataframe['building_mean'] = test_dataframe['building_id'].map(building_mean)
    test_dataframe['building_median'] = test_dataframe['building_id'].map(building_median)
    test_dataframe['building_min'] = test_dataframe['building_id'].map(building_min)
    test_dataframe['building_max'] = test_dataframe['building_id'].map(building_max)
    test_dataframe['building_std'] = test_dataframe['building_id'].map(building_std)
    return train_dataframe, test_dataframe


def add_lag_feature(weather_df, window=3):
    group_df = weather_df.groupby('site_id')
    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
    rolled = group_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.max().reset_index().astype(np.float16)
    lag_min = rolled.min().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    lag_mean = lag_mean.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
    lag_max = lag_max.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
    lag_min = lag_min.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
    lag_std = lag_std.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
    for col in cols:
        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]
        weather_df[f'{col}_max_lag{window}'] = lag_max[col]
        weather_df[f'{col}_min_lag{window}'] = lag_min[col]
        weather_df[f'{col}_std_lag{window}'] = lag_std[col]
    return weather_df


def add_rolling_statistics(weather_dataframe:pd.DataFrame):
    weather_dataframe['cloud_coverage'] += 1
    weather_dataframe['precip_depth_1_hr'] += 1
    criteria_for_precip = (weather_dataframe['site_id'] == 1) | (weather_dataframe['site_id'] == 12) | (weather_dataframe['site_id'] == 5)
    weather_dataframe.loc[criteria_for_precip, 'precip_depth_1_hr'] = 0
    criteria_for_cloud = (weather_dataframe['site_id'] == 11) | (weather_dataframe['site_id'] == 7)
    weather_dataframe.loc[criteria_for_cloud, 'cloud_coverage'] = 0
    criteria_for_sea_level = (weather_dataframe['site_id'] == 5)
    weather_dataframe.loc[criteria_for_sea_level, 'sea_level_pressure'] = 0
    weather_dataframe = add_lag_feature(weather_dataframe, window=3)
    weather_dataframe = add_lag_feature(weather_dataframe, window=72)
    return weather_dataframe


if __name__ == '__main__':
    train_df = pd.read_pickle(path=cfg.df_data_dir + "train_df.pkl")
    weather_train_df = pd.read_pickle(path=cfg.df_data_dir + "weather_train_df.pkl")
    test_df = pd.read_pickle(path=cfg.df_data_dir + "test_df.pkl")
    weather_test_df = pd.read_pickle(path=cfg.df_data_dir + "weather_test_df.pkl")
    building_meta_df = pd.read_pickle(path=cfg.df_data_dir + "building_metadata_df.pkl")
    train_df, test_df = merge_building_data(train_df, test_df, building_meta_df)
    print("skewing meter_reading variable with log(1+p)")
    train_df['meter_reading'] = np.log1p(train_df['meter_reading'])
    print(" ----------------- Adding Target features related to building meter reading consumption-----------------")
    train_df, test_df = add_target_feature(train_df, test_df)
    print("train data frame shape : \t {}".format(train_df.shape))
    print("test data frame shape : \t {}".format(test_df.shape))
    print(" ----------------- Filling Missing values in weather df -----------------")
    weather = fill_missing_data(weather_train_df, weather_test_df)
    print(" ----------------- Merging weather with train and test -----------------")
    train_df, test_df = merge_weather_data(train_df, test_df, weather)
    print(" ----------------- Removing Outliers -----------------")
    train_df = remove_outliers(train_df)
    train_df['DataType'], test_df['DataType'] = 'train', 'test'
    total_df = pd.concat([train_df, test_df], ignore_index=True, axis=0, sort=False)
    print(" ----------------- Features Extraction -----------------")
    total_df = feature_extraction(total_df)
    print(" ----------------- Encode Categorical columns -----------------")
    le = LabelEncoder()
    total_df['primary_use'] = total_df['primary_use'].astype(str)
    total_df['primary_use'] = le.fit_transform(total_df['primary_use']).astype(np.int8)
    total_df['meter'] = le.fit_transform(total_df['meter']).astype(np.int8)
    print(" ----------------- Drop features -----------------")
    total_df = drop_features(total_df)
    print_missing_in_df(total_df)
    print(" ----------------- Divide to Test and Train -----------------")
    train_df = total_df[total_df['DataType'] == 'train']
    test_df = total_df[total_df['DataType'] == 'test']
    train_df.drop(['DataType'], axis='columns', inplace=True)
    test_df.drop(['DataType'], axis='columns', inplace=True)
    train_y = train_df['meter_reading']
    train_x = train_df.drop(['meter_reading', 'row_id'], axis='columns')
    test_df = test_df.drop(['meter_reading'], axis='columns')
    print("train data frame shape : \t {}".format(train_df.shape))
    print("test data frame shape : \t {}".format(test_df.shape))
    train_x.to_pickle(path=cfg.ready_dir + "train_X.pkl")
    train_y.to_pickle(path=cfg.ready_dir + "train_y.pkl")
    test_df.to_pickle(path=cfg.ready_dir + "test_X.pkl")
