import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommand_with_movies(df_main,n,movies):
    data_2=df_main.copy()
    # 영화 간 유사도 산출

    # 1. user_movie table
    user_movie_DF=data_2.pivot_table('rating','userId','title').fillna(0)
    movie_user_DF=user_movie_DF.transpose()
    movie_user_sim = cosine_similarity(movie_user_DF,movie_user_DF)
    movie_user_simDF = pd.DataFrame(movie_user_sim,
                                    index=movie_user_DF.index,
                                    columns=movie_user_DF.index)

    # ex
    sim_movies = list()
    for seen_movie in movies:
        temp = movie_user_simDF.loc[movie_user_simDF.index != seen_movie,:][seen_movie].nlargest(100)
        sim_movies.append(temp)
    result=pd.DataFrame(pd.concat(sim_movies).groupby('title').sum().nlargest(n))
    return result