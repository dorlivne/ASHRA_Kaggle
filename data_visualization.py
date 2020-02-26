import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import CONFIG as cfg
import seaborn as sns
FILES = ['test.csv', 'building_metadata.csv', 'train.csv', 'weather_test.csv', 'sample_submission.csv', 'weather_train.csv']
METER_DIC = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}
METER_COLOR = {0: 'orange', 1: 'red', 2: 'blue', 3: 'pink'}

# check distribution of consecutive NA parts in data
def plot_series_and_consequtive_nans(
        df_weather_train: pd.DataFrame,
        df_weather_test: pd.DataFrame,
        column: str,
  #      site_id: int,
  #      index_slice: slice = None
):
    """
    Estimates consecutive NA blocks and plots interactive timeseries with missing data
    If slice is passed - perform that steps for selected data slice only
    clips upper block length at 24 (hours)
    """
    weather = pd.concat(
        [
            df_weather_train,
            df_weather_test
        ],
        ignore_index=True
    )
    unique_site_ids = sorted(np.unique(weather['site_id']))
    weather = weather.set_index(['site_id', 'timestamp'], drop=False).sort_index()

    # construct full index w/o missing dates
    full_index = pd.MultiIndex.from_product(
        [
            unique_site_ids,
            pd.date_range(start='2016-01-01 00:00:00', end='2018-12-31 23:00:00', freq='H')
        ]
    )

    print(f'init shape: {weather.shape}')
    weather = weather.reindex(full_index)
    print(f'full shape: {weather.shape}')

    weather['site_id'] = weather.index.get_level_values(0).astype(np.uint8)
    weather['timestamp'] = weather.index.get_level_values(1)
    fig, axes = plt.subplots(8, 2, figsize=(14, 30), dpi=100)
    fig2, axes2 = plt.subplots(8, 2, figsize=(14, 30), dpi=100)
    for i in range(weather['site_id'].nunique()):
        st = '2017-08-01 00:00:00'
        end = '2018-04-01 00:00:00'
        index_slice = slice(st, end)
        # series = df[df['site_id'] == i][[column, 'timestamp']].copy()
        series = weather.loc[i][column].copy()
        series = series.loc[index_slice]
        # define consecutive nan intervals
        nan_intervals = series.isnull().astype(int).groupby(
            series.notnull().astype(int).cumsum()).sum().clip(0, 24).value_counts()
        nan_intervals = nan_intervals[nan_intervals.index != 0]
        try:
            nan_intervals.plot(kind='bar', ax=axes2[i % 8][i // 8], rot=0).set_xlabel('block length, n points')
            axes2[i % 8][i // 8].set_title(f'consecutive NaNs in site "{i}" for column "{column}": {nan_intervals.sum()}')
        except TypeError:
            pass
        # to show missing values as simple interpolations
        interpolated = series.interpolate()
        to_plot = pd.DataFrame({column: series, 'missing': interpolated})
        to_plot.loc[~to_plot[column].isnull(), 'missing'] = np.nan
        to_plot.plot(ax=axes[i % 8][i // 8]).set_xlabel('timestamp')
        axes[i % 8][i // 8].set_ylabel(f'{column}')
        axes[i % 8][i // 8].set_title(f'site "{i}", timeseries for column "{column}"')
        # to_plot.iplot(
        #     dimensions=(240 * 3, 320),
        #     title=f'site "{i}", timeseries for column "{column}"',
        #     xTitle='timestamp',
        #     yTitle=f'{column}', ax=axes[i % 8][i // 8]
        # )
    fig.show()
    fig2.show()
    fig.savefig(cfg.eda_dir + "/missing sequences.png")
    fig2.savefig(cfg.eda_dir + "/missing sequences bar.png")

# def reduce_mem_usage(df, verbose=True):
#     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#     start_mem = df.memory_usage().sum() / 1024**2
#     for col in df.columns:
#         col_type = df[col].dtypes
#         if col_type in numerics:
#             c_min = df[col].min()
#             c_max = df[col].max()
#             if str(col_type)[:3] == 'int':
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     df[col] = df[col].astype(np.int64)
#             else:
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     df[col] = df[col].astype(np.float16)
#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     df[col] = df[col].astype(np.float64)
#     end_mem = df.memory_usage().sum() / 1024**2
#     if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
#     return df
#
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
    train_df = pd.read_csv(root_dir + 'train.csv', parse_dates=['timestamp'])
    weather_train_df = pd.read_csv(root_dir + 'weather_train.csv', parse_dates=['timestamp'])
    test_df = pd.read_csv(root_dir + 'test.csv', parse_dates=['timestamp'])
    weather_test_df = pd.read_csv(root_dir + 'weather_test.csv', parse_dates=['timestamp'])
    building_meta_df = pd.read_csv(root_dir + 'building_metadata.csv')
    # train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    # test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
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


def load_combined(data_dir=cfg.combined_data_dir):
    root_dir = data_dir
    train_df = pd.read_pickle(root_dir + 'train_df.pkl')
    test_df = pd.read_pickle(root_dir + 'test_df.pkl')
    return train_df, test_df


def merge_building_data(train_df, test_df, building_meta_df):
    building_meta_df['year_built'] = building_meta_df['year_built'].fillna(1969)  # mode
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

def merge_weather_data(train, test, weather):
    temp_df = train[['site_id', 'timestamp']]
    temp_df = temp_df.merge(weather, on=['site_id', 'timestamp'], how='left')

    del temp_df['site_id'], temp_df['timestamp']
    train = pd.concat([train, temp_df], axis=1)

    temp_df = test[['site_id', 'timestamp']]
    temp_df = temp_df.merge(weather, on=['site_id', 'timestamp'], how='left')

    del temp_df['site_id'], temp_df['timestamp']
    test = pd.concat([test, temp_df], axis=1)

    return train, test


def plot_target_vs_meter_with_accordance_to_site(df: pd.DataFrame):
    fig, axes = plt.subplots(8, 2, figsize=(14, 30), dpi=100)
    for i in range(df['site_id'].nunique()):
        for meter_id in df[df['site_id'] == i]['meter'].value_counts(dropna=False).index.to_list():
            df[(df['site_id'] == i) & (df['meter'] == meter_id)][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()[
                'meter_reading'].plot(ax=axes[i % 8][i // 8], alpha=1, label='{}'.format(METER_DIC[meter_id]),
                                      color='tab:{}'.format(METER_COLOR[meter_id])).set_xlabel('')
            axes[i % 8][i // 8].legend()
            axes[i % 8][i // 8].set_title('site_id {}'.format(i), fontsize=13)
            plt.subplots_adjust(hspace=0.45)
    fig.savefig(cfg.eda_dir + "meter_measures_according_to_site.png")
    fig.show()


def plot_according_to_building_id(df: pd.DataFrame, meter_id, site_id, primary_use):
    range_of_graphs = df[(df['meter'] == meter_id) & (df['site_id'] == site_id) & (df['primary_use'] == primary_use)]['building_id']\
        .value_counts(dropna=False).index.to_list()
    num_of_graphs = len(range_of_graphs)
    tmp_num_for_plot = int((np.ceil(num_of_graphs / 2)))
    fig, axes = plt.subplots(tmp_num_for_plot, 2, figsize=(14, 36), dpi=100)
    for i, name in enumerate(range_of_graphs):
        try:
            df[(df['site_id'] == site_id) & (df['meter'] == meter_id) & (df['primary_use'] == primary_use) &
               (df['building_id'] == name)][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()[
                'meter_reading'].plot(ax=axes[i % tmp_num_for_plot][i // tmp_num_for_plot], alpha=1, label='{}'.format(METER_DIC[meter_id]),
                                      color='tab:{}'.format(METER_COLOR[meter_id])).set_xlabel('')
            axes[i % tmp_num_for_plot][i // tmp_num_for_plot].legend()
        except TypeError:
            pass
        axes[i % tmp_num_for_plot][i // tmp_num_for_plot].set_title('buliding_{}'.format(name), fontsize=13)
        plt.subplots_adjust(hspace=0.45)
    fig.savefig(cfg.eda_dir + "meter_measures_in_site_{}_with_meter_{}_{}_buildings.png".format(site_id, meter_id, primary_use))
    fig.show()

def plot_target_vs_specific_meter_with_accordance_to_specific_site_according_to_variable(df: pd.DataFrame, meter_id, site_id, var):
    range_of_graphs = df[(df['meter'] == meter_id) & (df['site_id'] == site_id)][var].value_counts().index.to_list()
    num_of_graphs = len(range_of_graphs)
    i = 0
    fig, axes = plt.subplots(num_of_graphs // 2, 2, figsize=(14, 30), dpi=100)
    for name in range_of_graphs:
        try:
            df[(df['site_id'] == site_id) & (df['meter'] == meter_id) & (df[var] == name)][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()[
                'meter_reading'].plot(ax=axes[i % 8][i // 8], alpha=1, label='{}'.format(METER_DIC[meter_id]),
                                      color='tab:{}'.format(METER_COLOR[meter_id])).set_xlabel('')
            axes[i % num_of_graphs // 2][i // num_of_graphs // 2].legend()
        except TypeError:
            pass
        axes[i % (num_of_graphs // 2)][i // (num_of_graphs // 2)].set_title('{}:{}'.format(var, name), fontsize=13)
        plt.subplots_adjust(hspace=0.45)
        i += 1
    fig.savefig(cfg.eda_dir + "meter_measures_in_site_{}_with_meter_{}_according_to_{}.png".format(site_id, meter_id, var))
    fig.show()

def plot_target_vs_meter(df: pd.DataFrame):
    """
    plot that shows the measures according to each meter
    note: this plot includes all the buildings ID including the problematic once
          according to the the plot we can observe that according to steam meter we are given a weird result
          (very large and very spike with regards to the other meters)
    :param df: ASHRAE train data frame
    """
    fig, axes = plt.subplots(METER_DIC.__len__(), 1, figsize=(14, 18), dpi=100)
    for i in df['meter'].value_counts(dropna=False).index.to_list():
        name_of_meter = METER_DIC[i]
        df[df['meter'] == i][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(
            ax=axes[i], alpha=0.8, label='By hour', color='tab:blue').set_ylabel('Mean meter reading', fontsize=13)
        df[df['meter'] == i][
            ['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(
            ax=axes[i], alpha=1, label='By day', color='tab:orange').set_xlabel('')
        axes[i].legend()
        axes[i].set_title('Meter: {}'.format(name_of_meter), fontsize=13)
    fig.savefig(cfg.eda_dir + "meter_measures.png")
    fig.show()


def plot_with_and_without(train:pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(14, 20), dpi=100)

    train[(train['meter'] == 2)][['timestamp', 'meter_reading']].set_index(
        'timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[0], alpha=0.8, label='By hour',
                                                                color='tab:blue').set_ylabel('Mean meter reading',
                                                                                             fontsize=13)
    train[(train['meter'] == 2)][['timestamp', 'meter_reading']].set_index(
        'timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[0], alpha=1, label='By day',
                                                                color='tab:orange').set_xlabel('')

    train[~(train['building_id'] == 1099) & (train['meter'] == 2)][['timestamp', 'meter_reading']].set_index(
        'timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[1], alpha=0.8, label='By hour',
                                                                color='tab:blue').set_ylabel('Mean meter reading',
                                                                                             fontsize=13)
    train[~(train['building_id'] == 1099) & (train['meter'] == 2)][['timestamp', 'meter_reading']].set_index(
        'timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[1], alpha=1, label='By day',
                                                                color='tab:orange').set_xlabel('')

    axes[0].set_title('meter steam', fontsize=13)
    axes[1].set_title('meter steam with building_id 1099 excluded', fontsize=13)
    plt.subplots_adjust(hspace=0.45)
    fig.savefig(cfg.eda_dir + "steam_meter_measures_building_1099.png")
    fig.show()



def plot_meter_reading_boxplot(df:pd.DataFrame):
    fig, axes = plt.subplots(1, 1, figsize=(14, 6))
    sns.boxplot(x='site_id', y='year_built', data=df, showfliers=False)
    axes.set_title('site_id vs year_built test', fontsize=13)
    fig.savefig(cfg.eda_dir + "year_built_boxplots_test.png")
    # fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    # for i in df['meter'].value_counts(dropna=False).index.to_list():
    #     name_of_meter = METER_DIC[i]
    #     sns.boxplot(x='site_id', y='meter_reading', data=df[df['meter'] == i], showfliers=False, ax=axes[i % 2][i // 2])
    #     axes[i % 2][i // 2].set_title('meter : {}'.format(name_of_meter), fontsize=13)
    # fig.savefig(cfg.eda_dir + "measure_meter_boxplots_according_to_meter_type.png")
    # fig, axes = plt.subplots(1, 1, figsize=(14, 6))
    # sns.boxplot(x='site_id', y='air_temperature', data=df)
    # axes.set_title('temperature in each site', fontsize=13)
    # fig.savefig(cfg.eda_dir + "temp_in_site.png")
    # fig, axes = plt.subplots(1, 1, figsize=(14, 6))
    # sns.boxplot(x='site_id', y='meter_reading', data=df, showfliers=False)
    # fig.savefig(cfg.eda_dir + "meter_reading_in_site_box_plot.png")


def plot_meter_reading_according_to_buildings(df:pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    for i in df['meter'].value_counts(dropna=False).index.to_list():
        name_of_meter = METER_DIC[i]
        df[df['meter'] == i].groupby('building_id')['meter_reading'].mean().plot(ax=axes[i % 2][i // 2])
        axes[i % 2][i // 2].set_title('Mean meter {} reading by building_id'.format(name_of_meter), fontsize=14)
        axes[i % 2][i // 2].set_ylabel('Mean meter {}'.format(name_of_meter), fontsize=14)
    fig.savefig(cfg.eda_dir + "building_outlier.png")
    fig.show()


def plot_measurements_vs_variable(df:pd.DataFrame, measurments='meter_reading', variable='floor_count'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    for i in df['meter'].value_counts(dropna=False).index.to_list():
        if i == 0:
            continue
        name_of_meter = METER_DIC[i]
        pd.Series(index=df[df['meter'] == i][variable].value_counts().index,
                  data=df[df['meter'] == i].groupby(variable)[measurments].transform(
                      'mean').value_counts().index).sort_index().plot(kind='bar', rot=0, ax=axes[i % 2][i // 2])
        axes[i % 2][i // 2].set_xlabel(variable)
        axes[i % 2][i // 2].set_ylabel('Mean {} {}'.format(name_of_meter, measurments))
        axes[i % 2][i // 2].set_title('Mean {} {} by {}'.format(name_of_meter, measurments, variable))
    fig.savefig(cfg.eda_dir + "meter_{}_vs_{}.png".format(measurments, variable))
    fig.show()



def plot_meters_in_buildings(train:pd.DataFrame, test: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    train.groupby('building_id')['meter'].nunique().plot(style='.', ax=axes[0])
    test.groupby('building_id')['meter'].nunique().plot(style='.', ax=axes[1])
    axes[0].set_title('Train', fontsize=14)
    axes[1].set_title('Test', fontsize=14)
    axes[0].set_xlabel('building_id')
    axes[1].set_xlabel('building_id')
    axes[0].set_ylabel('Number of meters installed')
    plt.yticks([1, 2, 3, 4])
    fig.suptitle('Number of meters installed per building_id', fontsize=16)
    fig.savefig(cfg.eda_dir + "meter_per_building.png")
    fig.show()

def plot_col_vs_time_according_to_site(train, test, col):
    fig, axes = plt.subplots(8, 2, figsize=(14, 30), dpi=100)
    for i in range(train['site_id'].nunique()):
        train[train['site_id'] == i][['timestamp', col]].set_index('timestamp').resample('H').mean()[
            col].plot(ax=axes[i % 8][i // 8], alpha=0.8, label='By hour', color='tab:blue').set_ylabel(
            'Mean {}'.format(col), fontsize=13)
        test[test['site_id'] == i][['timestamp',col]].set_index('timestamp').resample('H').mean()[
            col].plot(ax=axes[i % 8][i // 8], alpha=0.8, color='tab:blue', label='').set_xlabel('')
        train[train['site_id'] == i][['timestamp', col]].set_index('timestamp').resample('D').mean()[
            col].plot(ax=axes[i % 8][i // 8], alpha=1, label='By day', color='tab:orange')
        test[test['site_id'] == i][['timestamp', col]].set_index('timestamp').resample('D').mean()[
            col].plot(ax=axes[i % 8][i // 8], alpha=1, color='tab:orange', label='').set_xlabel('')
        axes[i % 8][i // 8].legend()
        axes[i % 8][i // 8].set_title('site_id {}'.format(i), fontsize=13)
        axes[i % 8][i // 8].axvspan(test['timestamp'].min(), test['timestamp'].max(), facecolor='green', alpha=0.2)
        plt.subplots_adjust(hspace=0.45)
    fig.savefig(cfg.eda_dir + "{}_in_each_site_according_time.png".format(col))
    fig.show()


def plot_cloud_coverage_vs_meter_measurements(train):
    fig, axes = plt.subplots(2, 2, figsize=(14, 6), dpi=100)
    for i in train['meter'].value_counts(dropna=False).index.to_list():
        name_of_meter = METER_DIC[i]
        train[train['meter'] == i]['cloud_coverage'].value_counts(dropna=False, normalize=True).sort_index()\
            .plot(kind='bar', rot=0, ax=axes[i % 2][i // 2]).set_xlabel('cloud_coverage value')
        axes[i % 2][i // 2].set_title('Distribution in train for meter {}'.format(name_of_meter), fontsize=14)
        ax2 = axes[i % 2][i // 2].twinx()
        train[train['meter'] == i][['cloud_coverage', 'meter_reading']].replace(np.nan, 'nan').groupby('cloud_coverage')[
            'meter_reading'].mean().plot(ax=ax2, style='D-', grid=False, color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.set_ylabel('Mean meter reading', color='tab:orange', fontsize=14)
        ax2.set_xticklabels(train['cloud_coverage'].value_counts(dropna=False).sort_index().index)
        plt.subplots_adjust(wspace=0.4)
    fig.savefig(cfg.eda_dir + "meter_measurements_vs_cloud_coverage.png")
    fig.show()


def plot_meter_measurements_according_to_hour(df:pd.DataFrame, resolution='hour'):
    if resolution == 'hour':
       df[resolution] = df['timestamp'].dt.hour

    if resolution == 'day':
        df[resolution] = df['timestamp'].dt.day
    if resolution == 'month':
        df[resolution] = df['timestamp'].dt.month
    data = df[resolution].value_counts(dropna=False, normalize=True).sort_index().values
    len_res = np.arange(len(data))
    width = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(14, 6), dpi=100)
    for i in df['meter'].value_counts(dropna=False).index.to_list():
        name_of_meter = METER_DIC[i]
        axes[i % 2][i // 2].plot(len_res, df[df['meter'] == i][[resolution,'meter_reading']].
                                 groupby(resolution)['meter_reading'].mean().sort_index().values, 'D-',
                                 color='tab:{}'.format(METER_COLOR[i]), label='mean meter {} reading'.format(name_of_meter))
        axes[i % 2][i // 2].set_ylabel('Meter {} measurments'.format(name_of_meter))
        axes[i % 2][i // 2].set_xlabel(resolution)
    fig.savefig(cfg.eda_dir + "meter_measurements_vs_{}.png".format(resolution))
    fig.show()


def plot_floor_vs_sqr_feet(train: pd.DataFrame, test:pd.DataFrame):
    fig, axes = plt.subplots(8, 2, figsize=(14, 30), dpi=100)
    for i in range(train['site_id'].nunique()):
        try:
            train[train['site_id'] == i].groupby('floor_count')['square_feet'].mean().\
                plot(ax=axes[i % 8][i // 8], color='tab:blue', label='train').set_ylabel('Mean Sqr feet')
            axes[i % 8][i // 8].set_xlabel('floor count')
            test[test['site_id'] == i].groupby('floor_count')['square_feet'].mean().\
                plot(ax=axes[i % 8][i // 8], color='tab:red', label='test').set_ylabel('Mean Sqr feet')
            axes[i % 8][i // 8].set_xlabel('floor count')
        except TypeError:
            pass
        axes[i % 8][i // 8].set_title('floor_Vs_sqr_in_site_{}'.format(i))
        plt.subplots_adjust(hspace=0.45)
    fig.savefig(cfg.eda_dir + "floor_vs_sqr_feet.png")
    fig.show()


def plot_time_miss_adjustment(train:pd.DataFrame, test:pd.DataFrame, title='after'):
    fig, axes = plt.subplots(8, 2, figsize=(14, 30), dpi=100)
    tmp_df = pd.concat([train, test], ignore_index=True, axis=0, sort=False)
    tmp_df['hour'] = tmp_df['timestamp'].dt.hour
    for i in range(tmp_df['site_id'].nunique()):
        tmp_df[tmp_df['site_id'] == i].groupby('hour')['air_temperature'].mean().\
            plot(ax=axes[i % 8][i // 8], color='tab:blue').set_ylabel('Mean air temperature')
        axes[i % 8][i // 8].set_xlabel('hour')
        axes[i % 8][i // 8].set_title('air_temperature in site {}'.format(i))
        plt.subplots_adjust(hspace=0.45)
    # fig.savefig(cfg.eda_dir + "temp in site according to hour_{}_adjustement_pic.png".format(title))
    fig.show()


def fix_time_adjustment(weather_train:pd.DataFrame, weather_test:pd.DataFrame):
    """
    adjusting the hours of each sample to match night and day
    :param weather_train:
    :param weather_test:
    :return:
    """
    # weather_train['DataType'], weather_test['DataType'] = 'train', 'test'
    weather_key = ['site_id', 'timestamp']
    weather = pd.concat([weather_train, weather_test], ignore_index=True, axis=0, sort=False)
    temp_skeleton = weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()
    data_to_plot = temp_skeleton.copy()
    data_to_plot["hour"] = data_to_plot["timestamp"].dt.hour
    count = 1
    fig = plt.figure(figsize=(25, 15))
    for site_id, data_by_site in data_to_plot.groupby('site_id'):
        by_site_by_hour = data_by_site.groupby('hour').mean()
        ax = plt.subplot(4, 4, count)
        plt.plot(by_site_by_hour.index, by_site_by_hour['air_temperature'], 'xb-')
        ax.set_title('site: ' + str(site_id))
        count += 1
    plt.tight_layout()
    plt.show()
    # fig.savefig(cfg.eda_dir + "/air_temperature_before_Adjustment.png")
    del data_to_plot
    temp_skeleton['temp_rank'] = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.date])[
        'air_temperature'].rank('average')
    df_2d = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.hour])['temp_rank'].mean().unstack(level=1)
    site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)
    site_ids_offsets.index.name = 'site_id'

    def timestamp_align(df):
        df['offset'] = df.site_id.map(site_ids_offsets)
        df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))
        df['timestamp'] = df['timestamp_aligned']
        del df['timestamp_aligned'], df['offset']
        return df

    return timestamp_align(weather_train), timestamp_align(weather_test)


def plot_num_of_samples_in_category(category, df:pd.DataFrame):
    fig = plt.figure(figsize=(8, 6))
    df[category].value_counts(normalize=True).sort_values().plot(kind='bar')
    plt.title("Count of {} Variable".format(category))
    plt.xlabel(category)
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    fig.savefig(cfg.eda_dir + "samples_in_{}.png".format(category))
    fig.show()


def plot_meter_reading_dist(df:pd.DataFrame):
    fig = plt.figure()
    sns.distplot(np.log1p(df['meter_reading']), kde=False)
    fig.savefig(cfg.eda_dir + "meter reading dist")
    fig.show()


def plot_primary_use_Vs_meter(df:pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 6), dpi=100)
    for i in df['meter'].value_counts(dropna=False).index.to_list():
        name_of_meter = METER_DIC[i]
        df[df['meter'] == i].groupby('primary_use')['meter_reading'].nunique().plot(kind='bar', ax=axes[i % 2][i // 2])
        axes[i % 2][i // 2].set_ylabel('Meter {} measurements'.format(name_of_meter))
        axes[i % 2][i // 2].set_xlabel('Primary use')
    fig.savefig(cfg.eda_dir + "meter_measurements_vs_primary_use.png")
    fig.show()


def temp_plot(df:pd.DataFrame):
    count = 1
    df['hour'] = df.timestamp.dt.hour
    fig = plt.figure(figsize=(25, 50))
    for site_id, data_by_site in df[df['meter'] == 0].groupby('site_id'):
        by_site_by_hour = data_by_site.groupby('hour').mean()
        ax = plt.subplot(15, 4, count)
        plt.plot(by_site_by_hour.index, by_site_by_hour['meter_reading'], 'xb-')
        ax.set_title('site: ' + str(site_id))
        count += 1
    plt.tight_layout()
    fig.show()
    fig.savefig(cfg.eda_dir + "site_vs_meter_reading.png")


def plot_corrs(df:pd.DataFrame):
    corrupted_data_idx = (
            (df.meter == 0)
            & (df.site_id == 0)
            & (df.timestamp.dt.dayofyear < 140))
    df = df[~corrupted_data_idx]
    correlations_init = dict()
    df['meter_reading'] = np.log1p(df['meter_reading'])
    groups = df[['meter', 'site_id']].drop_duplicates().values.tolist()
    groups = list(tuple(e) for e in groups)
    for (m, sid) in groups:
        idx = (df.meter == m) & (df.site_id == sid)
        corrs = df.loc[idx, ['air_temperature']].corrwith(df.loc[idx, 'meter_reading'], method='spearman')
        correlations_init[(m, sid)] = dict(corrs)

    # create dataframe from it
    df_corr_init = pd.DataFrame(correlations_init).T.sort_index()
    df_corr_init.index = df_corr_init.index.set_names(['meter', 'site_id'])
    df_corr_init = df_corr_init.unstack(level=[0])
    df_corr_init.style.highlight_null().format("{:.2%}")
    print(df_corr_init)
    return df_corr_init


if __name__ == '__main__':
    # # # # # ## need to run only once
    # train_df, weather_train_df, test_df, weather_test_df,\
    # building_meta_df = extract_data()
    # print(train_df.shape)
    # print(test_df.shape)
    # train_df = reduce_mem_usage(train_df)
    # weather_train_df = reduce_mem_usage(weather_train_df)
    # test_df = reduce_mem_usage(test_df)
    # weather_test_df = reduce_mem_usage(weather_test_df)
    # building_meta_df = reduce_mem_usage(building_meta_df)
    # print(train_df.shape)
    # print(test_df.shape)
    # train_df.to_pickle(path=cfg.df_data_dir + "train_df.pkl")
    # weather_train_df.to_pickle(path=cfg.df_data_dir + "weather_train_df.pkl")
    # test_df.to_pickle(path=cfg.df_data_dir + "test_df.pkl")
    # weather_test_df.to_pickle(path=cfg.df_data_dir + "weather_test_df.pkl")
    # building_meta_df.to_pickle(path=cfg.df_data_dir + "building_metadata_df.pkl")
    # train_df, test_df = merge_building_data(train_df, test_df, building_meta_df)
    # print(train_df.shape)
    # print(test_df.shape)
    # temp_plot(train_df)
    # print(train_df.shape)
    # print(test_df.shape)
    # # the above is commented out because it only needs to run once
    # train_df, test_df = load_combined()
    # train_df, test_df = merge_weather_data(train_df, test_df, weather_train_df, weather_test_df)
    weather_train_df = pd.read_pickle(path=cfg.df_data_dir + "weather_train_df.pkl")
    weather_test_df = pd.read_pickle(path=cfg.df_data_dir + "weather_test_df.pkl")
    plot_col_vs_time_according_to_site(weather_train_df, weather_test_df, 'cloud_coverage')
    plot_col_vs_time_according_to_site(weather_train_df, weather_test_df, 'sea_level_pressure')
    plot_col_vs_time_according_to_site(weather_train_df, weather_test_df, 'wind_direction')
    plot_col_vs_time_according_to_site(weather_train_df, weather_test_df, 'precip_depth_1_hr')
    plot_col_vs_time_according_to_site(weather_train_df, weather_test_df, 'wind_speed')
    # plot_series_and_consequtive_nans(weather_train, weather_test, 'air_temperature')
    # train_df.drop(['offset'], inplace=True, axis='columns')
    # test_df.drop(['offset'], inplace=True, axis='columns')
    # temp_plot(train_df)
    # plot_time_miss_adjustment(train_df, test_df, title='after')
    # plot_primary_use_Vs_meter(train_df)
    # plot_meter_reading_dist(train_df)
    # plot_num_of_samples_in_category('primary_use', building_meta_df)
    # print("train:\n")
    # for col in train_df.columns:
    #     print(col)
    # print("test:\n")
    # for col in test_df.columns:
    #     print(col)
    # print(train_df.shape)
    # print(test_df.shape)
    # train_df.to_pickle(path=cfg.combined_data_dir + "train_df.pkl")
    # test_df.to_pickle(path=cfg.combined_data_dir + "test_df.pkl")

    # plot_floor_vs_sqr_feet(train_df, test_df)
    ###### building 1099 has a bad steam meter reading---proof is here:  ######
    # plot_target_vs_meter(train_df)
    # plot_target_vs_meter_with_accordance_to_site(train_df)
    # plot_target_vs_specific_meter_with_accordance_to_specific_site_according_to_variable(df=train_df, meter_id=2, site_id=13,var='primary_use')
    # plot_according_to_building_id(df=train_df, meter_id=2, site_id=13, primary_use='Education')
    # plot_with_and_without(train_df)
    # plot_meter_reading_boxplot(train_df)
    # plot_meter_reading_according_to_buildings(train_df)
    # plot_meter_measurements_vs_floor_count(train_df)
    # plot_meters_in_buildings(train_df, test_df)
    # plot_cloud_coverage_vs_meter_measurements(train_df)
    # plot_meter_measurements_according_to_hour(train_df, 'month')