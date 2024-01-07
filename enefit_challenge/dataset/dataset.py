import numpy as np 
import pandas as pd 

from pathlib import Path


class EnefitDataset():
    """
    Dataset Class for the Enefit Kaggle Challenge, 
    containing a collection of preprocessing methods
    """

    def __init__(self):
        self.data_path = Path('../../input/')
        # dictionary for feature aggregation over county-datetime
        self.weather_agg = {
            'temperature' : ['min', 'mean', 'max', 'std'],
            'dewpoint' : ['min', 'mean', 'max', 'std'],
            'cloudcover_high' : ['min', 'mean', 'max', 'std'],
            'cloudcover_low' : ['min', 'mean', 'max', 'std'],
            'cloudcover_mid' : ['min', 'mean', 'max', 'std'],
            'cloudcover_total' : ['min', 'mean', 'max', 'std'],
            '10_metre_u_wind_component' : ['min', 'mean', 'max', 'std'],
            '10_metre_v_wind_component' : ['min', 'mean', 'max', 'std'],
            'direct_solar_radiation' : ['min', 'mean', 'max', 'std'],
            'surface_solar_radiation_downwards' : ['min', 'mean', 'max', 'std'],
            'snowfall' : ['min', 'mean', 'max', 'std'],
            'total_precipitation' : ['min', 'mean', 'max', 'std'],
        }


    def load_enefit_training_data(self) -> pd.DataFrame:
        """
        Loads the enefit challenge training dataset, extracts time series features 
        from it and creates lags for the target variable. 
        """
        df_train = pd.read_csv(self.data_path / 'train.csv', parse_dates=['datetime'])
        df_train = df_train.dropna(how='any') # what to do with nans?
        df_train.sort_values(by="datetime", inplace=True)
        
        self.extract_dt_attributes(df_train, col="datetime")
        # daily + weekly lags
        df_train = self.create_lagged_features(df=df_train, lag=1)
        df_train = self.create_lagged_features(df=df_train, lag=7)
        
        df_client = pd.read_csv(self.data_path /'client.csv')

        location = pd.read_csv(self.data_path / "county_lon_lats.csv").drop(
            columns = ["Unnamed: 0"]
        )
        for k in ['latitude', 'longitude'] :
            location[k] = (10*location[k]).astype(int)

        df_electricity = pd.read_csv(self.data_path / 'electricity_prices.csv')
        df_electricity["forecast_date"] = pd.to_datetime(df_electricity["forecast_date"])
        df_electricity["origin_date"] = pd.to_datetime(df_electricity["origin_date"])
        df_electricity['time'] = df_electricity['forecast_date'].dt.strftime('%H:%M:%S')

        df_gas = pd.read_csv(self.data_path /'gas_prices.csv')
        df_gas["forecast_date"] = pd.to_datetime(df_gas["forecast_date"])
        df_gas["origin_date"] = pd.to_datetime(df_gas["origin_date"])

        df_weather_fc = pd.read_csv(
            self.data_path / 'forecast_weather.csv', 
            parse_dates=['origin_datetime', 'forecast_datetime']
        )
        df_weather_fc['forecast_datetime'] = df_weather_fc['forecast_datetime'].dt.tz_convert(None)
        df_weather_fc = self.get_county_loc(df_weather_fc, location)
        df_weather_fc = df_weather_fc.groupby(['county', 'forecast_datetime']).agg(self.weather_agg).reset_index()
        df_weather_fc.columns = ['_'.join([xx for xx in x if len(xx)>0]) for x in df_weather_fc.columns]
        df_weather_fc.columns = [x + '_f' if x not in ['county', 'forecast_datetime'] else x for x in df_weather_fc.columns]

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

        # merge with weather forecast data
        df_train = df_train.merge(
            df_weather_fc.rename(columns = {'forecast_datetime' : 'datetime'}),
            how='left',
            on=['county', 'datetime'],
        )

        return df_train

    def create_lagged_features(self, df: pd.DataFrame, lag: int) -> pd.DataFrame:
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

    def extract_dt_attributes(self, df: pd.DataFrame, col: str):
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

    def get_county_loc(self, h:pd.DataFrame, location: pd.DataFrame) -> pd.DataFrame:
        """
        Maps Latitude and Longitude of the h dataframe to the county code
        """
        h = h.drop_duplicates().reset_index(drop=True)
        for k in ['latitude', 'longitude'] :
            h[k] = (10*h[k]).astype(int)
        h = pd.merge(h, location, how='left', on=['latitude', 'longitude'])
        h['county'] = h['county'].fillna(-1).astype(int)
        return h

    def prepare_enefit_new_data(
            self,
            new_df: pd.DataFrame, 
            revealed_targets: pd.DataFrame, 
            df_client: pd.DataFrame, 
            df_electricity: pd.DataFrame, 
            df_gas: pd.DataFrame, 
            df_weather_fc: pd.DataFrame,
            train_df: pd.DataFrame
        )->pd.DataFrame:
        """
        Prepares for predictions new data coming from the Enefit Challenge, 
        taking as input DataFrames available at prediction time, extracting time series features from them
        and returning a unique dataframe.
        Then, it looks for previous lags for the target in the training set and merges 
        with the new dataset. 
        -------
        params:
        -------
        `new_df`: `pd.DataFrame` 
            new input data to perform predictions on
        `revealed_targets`: `pd.DataFrame` 
            contains the target column with a lag=2days
        `df_client`: `pd.DataFrame` 
            describes clients in `new_df`
        `df_electricity`: `pd.DataFrame` 
            contains price information for electricity
        `df_gas`: `pd.DataFrame` 
            contains price information for gas
        `df_weather_fc`: `pd.DataFrame`
            contains forecasts for weather features with a lag=1day
        `train_df`: `pd.DataFrame`
            the training dataset available for the Kaggle Enefit Challenge
        -------
        returns:
        -------
        `new_df`: `pd.DataFrame`
            the new data ready to be fed to the model
        """

        self.extract_dt_attributes(new_df, col="prediction_datetime")

        revealed_targets = revealed_targets.rename(columns={'target':'target_2_days_ago'})
        revealed_targets['prediction_datetime'] = pd.to_datetime(revealed_targets['datetime']) + pd.Timedelta(days=2)
        new_df = pd.merge(
            new_df,
            revealed_targets[
                [
                    'county', 'is_business','is_consumption','product_type', 
                    'prediction_datetime', 'target_2_days_ago', 'data_block_id'
                ]
            ],
            how='left',
            on=[
                'county', 'is_business','is_consumption','product_type', 
                'prediction_datetime', 'data_block_id'
            ]
        )

        df_electricity['forecast_date'] = pd.to_datetime(df_electricity['forecast_date'])
        df_electricity['time'] = df_electricity['forecast_date'].dt.strftime('%H:%M:%S')
        df_electricity['date'] = (df_electricity['forecast_date'] + pd.Timedelta(days=1)).dt.date
        new_df = pd.merge(
            new_df, 
            df_electricity[['time', 'date', 'euros_per_mwh']],
            how = 'left',
            on = ['time', 'date'] 
        )
        
        df_gas['date'] = (pd.to_datetime(df_gas['forecast_date']) + pd.Timedelta(days=1)).dt.date
        new_df = pd.merge(
            new_df, 
            df_gas[['date', 'lowest_price_per_mwh', 'highest_price_per_mwh']],
            how = 'left',
            on = ['date'] 
        )
        
        df_client['date'] = (pd.to_datetime(df_client['date']) + pd.Timedelta(days=2)).dt.date
        new_df = pd.merge(
            new_df, 
            df_client.drop("data_block_id", axis=1),
            how='left',
            on = ['date', 'product_type', 'county', 'is_business'],
        )
        
        new_df = new_df.rename(columns={'prediction_datetime':'datetime'})

        location = pd.read_csv(self.data_path / "county_lon_lats.csv").drop(
            columns = ["Unnamed: 0"]
        )
        for k in ['latitude', 'longitude'] :
            location[k] = (10*location[k]).astype(int)

        df_weather_fc['forecast_datetime'] = df_weather_fc['forecast_datetime'].dt.tz_convert(None)
        df_weather_fc = self.get_county_loc(df_weather_fc, location)
        df_weather_fc = df_weather_fc.groupby(['county', 'forecast_datetime']).agg(self.weather_agg).reset_index()
        df_weather_fc.columns = ['_'.join([xx for xx in x if len(xx)>0]) for x in df_weather_fc.columns]
        df_weather_fc.columns = [x + '_f' if x not in ['county', 'forecast_datetime'] else x for x in df_weather_fc.columns]

        new_df = new_df.merge(
            df_weather_fc.rename(columns = {'forecast_datetime' : 'datetime'}),
            how='left',
            on=['county', 'datetime'],
        )
        # identify matching obs in training set
        unique_identifiers = [
            "county", "is_business", "is_consumption", 
            "product_type", "datetime"
        ]
        matching_observation_columns = unique_identifiers + ["target_1_days_ago", "target_7_days_ago"]
        matching_observations = train_df[matching_observation_columns]
        new_df = pd.merge(
            new_df, 
            matching_observations, 
            on=unique_identifiers, 
            how="left", 
            suffixes=('', '_train')
        )

        return new_df