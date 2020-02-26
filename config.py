class CONFIG:
    data_dir = r'data/'
    df_data_dir = r'data_df/'
    combined_data_dir = r'combined_data_df/'
    eda_dir = r'visualizations/EDA/'
    ready_dir = r'ready_data/'
    results_df = r'results/results.pkl'
    results_csv =r'results/results.csv'
    feature_df = r'results/features_important.pkl'
    feature_dir = r'results/features_important_'
    model_dir = r'models/'
    MLP_dir = r'models/MLP/'
    RMLP_dir = r'models/RMLP/'
    leak_df = r'leak/leak.feather'
    test_feather = r'leak/test.feather'
    verbose = False
    folds = 5
    MLP_folds = 4
    timeout = 60
    lstm_units = 256
    batch_size = 1024
    int_cols = [
        # it's ordered but lives in integer scale
        'cloud_coverage', 'wind_direction']
    submission_list = ["submission_aggregated_LGBM.csv", "submission_aggregated_MLP.csv", "submission_aggregated_RMLP.csv"]

    @staticmethod
    def learning_rate_scheduler(epoch: int):
        if epoch <= 2:
            return 1e-2
        if 2 < epoch <= 4:
            return  5e-3
        if 4 < epoch <= 6:
            return 1e-3
        if 6 < epoch <= 7:
            return 0.0005
        if 7 < epoch <= 8:
            return 0.0001
        if 8 < epoch <= 10:
            return 0.00005

    @staticmethod
    def learning_rate_scheduler_RMLP(epoch: int):
        if epoch <= 10:
            return 1e-2
        if 10 < epoch <= 100:
            return 1e-3
        if 100 < epoch <= 200:
            return 5e-4
        if 200 < epoch <= 300:
            return 1e-4
        if 300 < epoch <= 350:
            return 5e-5
        if 350 < epoch:
            return 1e-6