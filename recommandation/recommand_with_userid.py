import pandas as pd
import datetime
from utils.get_data import get_data

from recommandation.recommand_general import recommand_general
from recommandation.recommand_with_genre import recommand_with_genres

from surprise import SVD
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def user_unseen_movie(df_main,user_id):
    result=df_main[~df_main['title'].isin(df_main[df_main['userId']==user_id]['title'].values)]
    return result['movieId'].unique()

def user_seen_movie(df_main,user_id):
    result=df_main[~df_main['title'].isin(df_main[df_main['userId']==user_id]['title'].values)]
    return result['movieId'].unique()

def recommand_user_id_svd(df_main,user_id):
    data = df_main.loc[:,['userId','movieId','rating']]
    reader=Reader() 
    s_data = Dataset.load_from_df(df=data,reader=reader)
    train,test  = train_test_split(s_data,test_size=1e-5)

    model = SVD()
    model.fit(train)
    movie_ids = user_unseen_movie(df_main,user_id)

    preds = list()
    for movie_id in movie_ids:
        preds.append(model.predict(uid=user_id,iid=movie_id,))

    id_title_mapper=df_main.loc[:,['movieId','title']].set_index('movieId').to_dict('dict')['title']
    temp=pd.DataFrame(preds).drop(columns=['r_ui','details','uid']).groupby('iid').mean('est').nlargest(200,'est').reset_index()
    result1=pd.concat([temp['iid'].map(id_title_mapper),temp['est']],axis=1)
    result1.rename(columns={'iid':'title'},inplace=True)
    return result1

# 영화 간 유사도 산출
def recommand_user_id_movie(df_main,user_id):
    seen_movies=user_seen_movie(df_main,user_id)
    data_2 = df_main.copy()
    user_movie_DF=data_2.pivot_table('rating','userId','title').fillna(0)
    movie_user_DF=user_movie_DF.transpose()
    movie_user_sim = cosine_similarity(movie_user_DF,movie_user_DF)
    movie_user_simDF = pd.DataFrame(movie_user_sim,
                                    index=movie_user_DF.index,
                                    columns=movie_user_DF.index)

    id_title_mapper=df_main.loc[:,['movieId','title']].set_index('movieId').to_dict('dict')['title']

    cosine_movie_sim = list()
    for seen_movie in seen_movies:
        movie_name = id_title_mapper[seen_movie]
        cosine_movie_sim.append(movie_user_simDF.loc[movie_user_simDF.index != movie_name,:][movie_name].nlargest(20))

    result2=pd.concat(cosine_movie_sim).reset_index().groupby('title').sum().nlargest(200,0)
    result2.reset_index(inplace=True)
    return result2

# 장르 유사성을 통한 필터링
def recommand_user_id_genres(df_main,user_id):
    seen_movies=user_seen_movie(df_main,user_id)
    seen_moviesDF=df_main.drop_duplicates('title')[df_main.drop_duplicates('title')['movieId'].isin(seen_movies)]
    genres=sum(seen_moviesDF['genres'].apply(lambda x : x.split('|')).values,[])
    
    genres = [i[0] for i in Counter(genres).most_common(2)]
    rc_general=recommand_general(df_main,0)

    data = df_main.copy()
    
    data_4genreDF=data.drop_duplicates('title').copy()
    data_4genreDF['genres']=data_4genreDF['genres'].apply(lambda x: x.split('|'))
    merged_recommand=pd.merge(rc_general,data_4genreDF.loc[:,['title','genres']],right_on='title',left_on='iid')
    result=merged_recommand[merged_recommand['genres'].apply(lambda x: True if len(set(genres)-set(x))==0 else False)].nlargest(200,'est').reset_index(drop=True)
    result3=result.loc[:,['title','est','genres']]
    return result3

def recommand_user_id_ensemble(df_main,user_id,n,weights=(0,0,0)):
    # id svd
    result1 = recommand_user_id_svd(df_main,user_id)
    result1['point'] = result1.index[::-1] * (1+weights[0])
    
    rc_general=recommand_general(df_main,0)
    est_mapper=rc_general.set_index('iid')['est'].to_dict()
    
    # id simmilar movies
    result2 = recommand_user_id_movie(df_main,user_id)
    result2['point'] = result2.index[::-1] * (1+weights[1])
    
    # id simmilar genres
    result3 = recommand_user_id_genres(df_main,user_id)
    result3['point'] = result3.index[::-1] * (1+weights[2])
    
    result=pd.concat(
        [
            result1.loc[:,['title','point']],
            result2.loc[:,['title','point']],
            result3.loc[:,['title','point']]
         ]
        )
    recommands=result.groupby('title').sum().nlargest(n,'point').reset_index()
    recommands['est']=recommands['title'].apply(lambda x : est_mapper[x])
    recommands=recommands.loc[:,['title','est','point']]
    
    result_1_mapper=result1.loc[:,['title','point']].set_index('title')['point'].to_dict()
    result_2_mapper=result2.loc[:,['title','point']].set_index('title')['point'].to_dict()
    result_3_mapper=result3.loc[:,['title','point']].set_index('title')['point'].to_dict()
    
    recommands['svd_est_point']=recommands.reset_index()['title'].map(result_1_mapper)
    recommands['movie_simillarity_point']=recommands.reset_index()['title'].map(result_2_mapper)
    recommands['genre_simillarity_point']=recommands.reset_index()['title'].map(result_3_mapper)
    
    return recommands.loc[:,['title','est','point','svd_est_point','movie_simillarity_point','genre_simillarity_point']]