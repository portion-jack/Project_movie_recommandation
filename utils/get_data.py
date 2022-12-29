import pandas as pd
import datetime

def get_data():
    # load data
    df_raw_1=pd.read_csv('data/ratings.csv')
    df_raw_2=pd.read_csv('data/movies.csv')

    # merge data
    df_merged=df_raw_1.merge(df_raw_2,on='movieId')

    # timestamp to datetime
    df_merged['time_line']=df_merged['timestamp'].apply(lambda x : datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d_%H:%m'))

    # get year and month data
    df_merged['year']=df_merged['time_line'].apply(lambda x: int(x.split('-')[0]))
    df_merged['month']=df_merged['time_line'].apply(lambda x: int(x.split('-')[1]))
    return df_merged
