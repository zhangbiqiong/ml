import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from gensim.models import Word2Vec
import datetime

df_movies = pd.read_csv("ml-latest-small/movies.csv")
df_ratings = pd.read_csv("ml-latest-small/ratings.csv")

movieId_to_name = pd.Series(
    df_movies.title.values, index=df_movies.movieId.values
).to_dict()
name_to_movieId = pd.Series(df_movies.movieId.values, index=df_movies.title).to_dict()

# Randomly display 5 records in the dataframe
for df in list((df_movies, df_ratings)):
    rand_idx = np.random.choice(len(df), 5, replace=False)
    display(df.iloc[rand_idx, :])
    print("Displaying 5 of the total " + str(len(df)) + " data points")


# fig = plt.figure()
# ax = fig.add_subplot()
# ax.set_title("Distribution of Movie Ratings", fontsize=16)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# plt.xlabel("Movie Rating", fontsize=14)
# plt.ylabel("Count", fontsize=14)

# plt.hist(df_ratings["rating"], color="#3F5D7D")

# plt.show()

from sklearn.model_selection import train_test_split

df_ratings_train, df_ratings_test= train_test_split(df_ratings,
                                                    stratify=df_ratings['userId'],
                                                    random_state = 15688,
                                                    test_size=0.30)

print("Number of training data: "+str(len(df_ratings_train)))
print("Number of test data: "+str(len(df_ratings_test)))


def rating_splitter(df):    
    df['liked'] = np.where(df['rating']>=4, 1, 0)
    df['movieId'] = df['movieId'].astype('str')
    gp_user_like = df.groupby(['liked', 'userId'])

    return ([gp_user_like.get_group(gp)['movieId'].tolist() for gp in gp_user_like.groups])

pd.options.mode.chained_assignment = None
splitted_movies = rating_splitter(df_ratings_train)


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
assert gensim.models.word2vec.FAST_VERSION > -1
import random

for movie_list in splitted_movies:
    random.shuffle(movie_list)

# from gensim.models import Word2Vec
# import datetime
# start = datetime.datetime.now()

# model = Word2Vec(sentences = splitted_movies, # We will supply the pre-processed list of moive lists to this parameter
#                  iter = 5, # epoch
#                  min_count = 10, # a movie has to appear more than 10 times to be keeped
#                  size = 200, # size of the hidden layer
#                  workers = 4, # specify the number of threads to be used for training
#                  sg = 1, # Defines the training algorithm. We will use skip-gram so 1 is chosen.
#                  hs = 0, # Set to 0, as we are applying negative sampling.
#                  negative = 5, # If > 0, negative sampling will be used. We will use a value of 5.
#                  window = 9999999)

# print("Time passed: " + str(datetime.datetime.now()-start))
# model.save('item2vec_20180327')

# from gensim.models import Word2Vec
# import datetime
# start = datetime.datetime.now()

# model_w2v_sg = Word2Vec(sentences = splitted_movies,
#                         iter = 10, # epoch
#                         min_count = 5, # a movie has to appear more than 5 times to be keeped
#                         size = 300, # size of the hidden layer
#                         workers = 4, # specify the number of threads to be used for training
#                         sg = 1,
#                         hs = 0,
#                         negative = 5,
#                         window = 9999999)

# print("Time passed: " + str(datetime.datetime.now()-start))
# model_w2v_sg.save('item2vec_word2vecSg_20180328')
# del model_w2v_sg


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import Word2Vec
model = Word2Vec.load('item2vec_20180327')
word_vectors = model.wv

import requests
import re
from bs4 import BeautifulSoup

def refine_search(search_term):
    """
    Refine the movie name to be recognized by the recommender
    Args:
        search_term (string): Search Term

    Returns:
        refined_term (string): a name that can be search in the dataset
    """
    refined_name=""
    target_url = "http://www.imdb.com/find?ref_=nv_sr_fn&q="+"+".join(search_term.split())+"&s=tt"
    html = requests.get(target_url).content
    parsed_html = BeautifulSoup(html, 'html.parser')
    for tag in parsed_html.find_all('td', class_="result_text"):
        search_result = re.findall('fn_tt_tt_1">(.*)</a>(.*)</td>', str(tag))
        if search_result:
            if search_result[0][0].split()[0]=="The":
                str_frac = " ".join(search_result[0][0].split()[1:])+", "+search_result[0][0].split()[0]
                refined_name = str_frac+" "+search_result[0][1].strip()
            else:
                refined_name = search_result[0][0]+" "+search_result[0][1].strip()
    return refined_name

def produce_list_of_movieId(list_of_movieName, useRefineSearch=False):
    """
    Turn a list of movie name into a list of movie ids. The movie names has to be exactly the same as they are in the dataset.
       Ambiguous movie names can be supplied if useRefineSearch is set to True
    
    Args:
        list_of_movieName (List): A list of movie names.
        useRefineSearch (boolean): Ambiguous movie names can be supplied if useRefineSearch is set to True

    Returns:
        list_of_movie_id (List of strings): A list of movie ids.
    """
    list_of_movie_id = []
    for movieName in list_of_movieName:
        if useRefineSearch:
            movieName = refine_search(movieName)
            print("Refined Name: "+movieName)
        if movieName in name_to_movieId.keys():
            list_of_movie_id.append(str(name_to_movieId[movieName]))
    return list_of_movie_id

def recommender(positive_list=None, negative_list=None, useRefineSearch=False, topn=20):
    recommend_movie_ls = []
    if positive_list:
        positive_list = produce_list_of_movieId(positive_list, useRefineSearch)
    if negative_list:
        negative_list = produce_list_of_movieId(negative_list, useRefineSearch)
    for movieId, prob in model.wv.most_similar_cosmul(positive=positive_list, negative=negative_list, topn=topn):
        recommend_movie_ls.append(movieId)
    return recommend_movie_ls

ls = recommender(positive_list=["Apollo 13 (1995)","Forrest Gump (1994)"], useRefineSearch=False, topn=5)
# print('Recommendation Result based on "Jumanji (1995)":')
display(df_movies[df_movies['movieId'].isin(ls)])
