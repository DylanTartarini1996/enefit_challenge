import numpy as np 
import pandas as pd 

from pathlib import Path
data_path = Path('../../input/')


def load_enefit_training_data() -> pd.DataFrame:
    """
    Loads the enefit challenge training dataset, extracts time series features 
    from it and creates lags for the target variable. 
    """
    df_train = pd.read_csv(data_path / 'train.csv', parse_dates=['datetime'])
    df_train = df_train.dropna(how='any') # what to do with nans?
    df_train.sort_values(by="datetime", inplace=True)
    extract_dt_attributes(df_train, col="datetime")
    df_train = create_lagged_features(df=df_train, lag=2)

    df_client = pd.read_csv(data_path /'client.csv')

    df_electricity = pd.read_csv(data_path / 'electricity_prices.csv')
    df_electricity["forecast_date"] = pd.to_datetime(df_electricity["forecast_date"])
    df_electricity["origin_date"] = pd.to_datetime(df_electricity["origin_date"])
    df_electricity['time'] = df_electricity['forecast_date'].dt.strftime('%H:%M:%S')

    df_gas = pd.read_csv(data_path /'gas_prices.csv')
    df_gas["forecast_date"] = pd.to_datetime(df_gas["forecast_date"])
    df_gas["origin_date"] = pd.to_datetime(df_gas["origin_date"])

    # merge with client data
    df_train = pd.merge(
        df_train,
        df_client.drop('date', axis=1),
        on = ['data_block_id', 'product_type', 'county', 'is_business'],
        how='left'
    )

    # merge with electricity and gas prices forecasts data
    df_train = pd.merge(
        df_train,
        df_electricity[['time', 'data_block_id', 'euros_per_mwh']],
        how = 'left',
        on = ['time', 'data_block_id'] 
    )
    df_train = pd.merge(
        df_train,
        df_gas[['data_block_id', 'lowest_price_per_mwh', 'highest_price_per_mwh']],
        how = 'left',
        on = ['data_block_id'] 
    )

    return df_train


def prepare_enefit_test_data(
        df: pd.DataFrame, 
        revealed_targets: pd.DataFrame, 
        df_client: pd.DataFrame, 
        df_electricity: pd.DataFrame, 
        df_gas: pd.DataFrame
    )->pd.DataFrame:
    """
    Loads the enefit challenge testing dataset, extracts time series features 
    from it and creates lags for the target variable
    """
    extract_dt_attributes(df)
    evealed_targets = revealed_targets.rename(columns={'target':'target_2_days_ago'})
    revealed_targets['datetime'] = pd.to_datetime(revealed_targets['datetime']) + pd.Timedelta(days=2)
    df = pd.merge(
        df,
        revealed_targets[
            ['county', 'is_business','is_consumption','product_type', 'datetime', 'target_2days_ago']
        ],
        how='left',
        on=['county', 'is_business','is_consumption','product_type', 'datetime']
    )
    df_electricity['forecast_date'] = pd.to_datetime(df_electricity['forecast_date'])
    df_electricity['time'] = df_electricity['forecast_date'].dt.strftime('%H:%M:%S')
    df_electricity['date'] = (df_electricity['forecast_date'] + pd.Timedelta(days=1)).dt.date
    df = pd.merge(
        df, 
        df_electricity[['time', 'date', 'euros_per_mwh']],
        how = 'left',
        on = ['time', 'date'] 
    )
    df_gas['date'] = (pd.to_datetime(df_gas['forecast_date']) + pd.Timedelta(days=1)).dt.date
    df = pd.merge(
        df, 
        df_gas[['date', 'lowest_price_per_mwh', 'highest_price_per_mwh']],
        how = 'left',
        on = ['date'] 
    )
    df_client['date'] = (df_client['date'] + pd.Timedelta(days=2)).dt.date
    df = pd.merge(
        df, 
        df_client,
        how='left',
        on = ['date', 'product_type', 'county', 'is_business'],
    )

    return df
    

def create_lagged_features(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    """
    Create lagged features by shifting data in a pandas DataFrame.

    This function adds lagged features to the input DataFrame 'df' by shifting 
    the `data_block_id` column by the specified `lag` number of days. 
    It then merges the original DataFrame with the shifted DataFrame
    to create new columns for lagged features.

    params:
    ---------
    `df`: `pd.DataFrame` 
        The input DataFrame.
    `lag`: `int` 
        The number of days to shift the 'data_block_id' column.
    """
    df['data_block_id_shifted'] = df['data_block_id'] + lag
    df = pd.merge(
        df, 
        (df[['county', 'is_business','is_consumption',
             'product_type','data_block_id_shifted', 'time', 'target']].rename(
                columns={
                    'data_block_id_shifted':'data_block_id', 
                    'target':f'target_{lag}_days_ago'
                    }
                )   
        ),
    on = ['county', 'is_business','is_consumption','product_type', 'data_block_id', 'time'], 
    how='left'
    )
    # drop the redundant column
    df.drop(columns=['data_block_id_shifted'],inplace=True)
    return df


def extract_dt_attributes(df: pd.DataFrame, col: str):
    """
    Extract different Time-Series attributes from a pandas DataFrame with a 
    datetime column.

    params:
    ---------
    `df`: `pd.DataFrame` 
        The input DataFrame.
    `col`: `str` 
        The datetime column to be used as source.
    """
    # convert datetime column, if not done already
    df[col] = pd.to_datetime(df[col])
    
    # dates and times
    df['date'] = df[col].dt.date
    df['time'] = df[col].dt.strftime('%H:%M:%S')
    
    #
    df['year'] = df[col].dt.year
    df['datediff_in_days'] = (
        df[col] - (df[col].min())
    ).dt.days
    
    # dictionary with time features as keys
    # and min and max as values
    time_features = {
        'hour': [0, 23],
        'dayofweek': [0, 6],
        'week': [1, 52],
        'month': [1, 12]
    }
    
    for c in time_features:
        if c=='week':
            df[c] = df[col].dt.isocalendar().week.astype(np.int32)
        else:
            df[c] = getattr(df[col].dt, c)
        
        ## sin and cosine features to capture the circular continuity
        col_min,col_max = time_features[c]
        angles = 2*np.pi*(df[c]-col_min)/(col_max-col_min+1)
        
        # add sin and cos
        df[c+'_sine'] = np.sin(angles).astype('float')
        df[c+'_cosine'] = np.cos(angles).astype('float')