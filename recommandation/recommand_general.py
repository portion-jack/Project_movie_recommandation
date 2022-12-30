import pandas as pd
# scikit_learn surprise
from surprise import SVD
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split

# builtin

def recommand_general(df_main,n):
    data_1 = df_main.loc[:,['userId','movieId','rating']]
    reader=Reader() 
    s_data = Dataset.load_from_df(df=data_1,reader=reader)
    train,test  = train_test_split(s_data,test_size=1e-5)
    model = SVD()
    model.fit(train)
    user_id = 'Undefined'
    movie_ids = data_1['movieId'].unique()
    preds = list()
    for movie_id in movie_ids:
        preds.append(model.predict(uid=user_id,iid=movie_id))
    id_title_mapper=df_main.drop_duplicates('title').loc[:,['movieId','title']].set_index('movieId').to_dict('dict')['title']
    if n ==0:
        temp=pd.DataFrame(preds).drop(columns=['r_ui','details','uid']).groupby('iid').mean('est').reset_index()
        result=pd.concat([temp['iid'].map(id_title_mapper),temp['est']],axis=1)
        return result
    else:
        temp=pd.DataFrame(preds).drop(columns=['r_ui','details','uid']).groupby('iid').mean('est').nlargest(n,'est').reset_index()
        result=pd.concat([temp['iid'].map(id_title_mapper),temp['est']],axis=1)
        return result.rename(columns={'iid':'title'})
