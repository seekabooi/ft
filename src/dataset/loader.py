import pandas as pd

def load_dataset(dataset_config):
    source = dataset_config['source']
    if source['type'] == 'file':
        path = source['params']['path']
        df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date').sort_index()

    elif source['type'] == 'api':
        from src.data_fetcher.api_source import ApiDataSource
        ds = ApiDataSource()
        # 将数据集配置中的 target_column 传入 params
        params = source['params'].copy()
        params['target_column'] = dataset_config['target_column']
        df = ds.fetch(**params)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date').sort_index()

    else:
        raise ValueError(f"Unsupported source type: {source['type']}")