import pandas as pd
import datetime
from utils.get_data import get_data
from recommandation.recommand_general import recommand_general

def recommand_with_genres(df_main,n,genres):
    rc_general=recommand_general(df_main,0)
    data = df_main.copy()
    data_4genreDF=data.drop_duplicates('title').copy()
    data_4genreDF['genres']=data_4genreDF['genres'].apply(lambda x: x.split('|'))
    merged_recommand=pd.merge(rc_general,data_4genreDF.loc[:,['title','genres']],right_on='title',left_on='iid')
    result=merged_recommand[merged_recommand['genres'].apply(lambda x: True if len(set(genres)-set(x))==0 else False)].nlargest(n,'est').reset_index(drop=True)
    return result.loc[:,['title','est','genres']]
