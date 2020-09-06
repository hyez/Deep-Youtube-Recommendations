from urllib.request import urlretrieve
import zipfile
import pandas as pd
import numpy as np
import random

import core.utils as utils
# from config import *

class Dataset(object):
	def __init__(self, dataset_type):
		users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
		self.users = pd.read_csv(
		'data/u.user', sep='|', names=users_cols, encoding='latin-1')

		watch_cols = ['user_id', 'movie_id', 'watch_hist_time']
		self.watches = pd.read_csv(
			'data/u.watch', sep='\t', names=watch_cols, encoding='latin-1')

		search_cols = ['user', 'search_hist']
		self.searches = pd.read_csv(
			'data/u.search', sep='\t', names=search_cols, encoding='latin-1')

		ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
		self.ratings = pd.read_csv(
			'data/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

		# The movies file contains a binary feature for each genre.
		genre_cols = [
			"genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
			"Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
			"Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
		]
		movies_cols = [
			'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
		] + genre_cols

		self.movies = pd.read_csv(
			'data/u.item', sep='|', names=movies_cols, encoding='latin-1')

		self.movies["year"] = self.movies['release_date'].apply(lambda x: str(x).split('-')[-1])

		self.movies['example_age'] = (pd.to_datetime("now") - pd.to_datetime(self.movies['release_date']))\
			/np.timedelta64(1,'D') 

		self.preprocess(dataset_type)


	def preprocess(self, dataset_type):
		self.movies["year"] = self.movies['release_date'].apply(lambda x: str(x).split('-')[-1])
		# example_age 추가
		self.movies['example_age'] = (pd.to_datetime("now") - pd.to_datetime(self.movies['release_date']))\
			/np.timedelta64(1,'D') 

		self.movies = utils.normalize_col(self.movies,'example_age')
		self.watches = utils.normalize_col(self.watches,'watch_hist_time')

		# data 합치기
		data = self.watches.merge(self.movies, on='movie_id').merge(self.users, on='user_id')
		data['user_id']=data['user_id'].astype(int)
		data['movie_id']=data['movie_id'].astype(int)
		data = data.set_index(['user_id']).sort_index()
		data = data.reset_index()
		data['movie_name'] = data['title'].str[:-6] # 년도 부분 자르기


		# occupation 인코딩
		occupations = data["occupation"].unique().tolist()
		occupations_encoded = {x: i for i, x in enumerate(occupations)}
		occupationsencoded2occupations = {i: x for i, x in enumerate(occupations)}

		# search history 인코딩
		search_hists = self.searches["search_hist"].unique().tolist()
		search_encoded = {x: i for i, x in enumerate(search_hists)}
		searchencoded2search = {i: x for i, x in enumerate(search_hists)}

		# 유저 인덱스 인코딩
		user_ids = data["user_id"].unique().tolist()
		user2user_encoded = {x: i for i, x in enumerate(user_ids)}
		userencoded2user = {i: x for i, x in enumerate(user_ids)}

		# 영화 인덱스 인코딩
		movie_ids = data["movie_id"].unique().tolist()
		movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
		movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

		# 영화 제목 인코딩
		title_ids = data["title"].unique().tolist()
		title2title_encoded = {x: i for i, x in enumerate(title_ids)}
		title_encoded2title = {i: x for i, x in enumerate(title_ids)}

		# 인코딩으로 바꾸기
		data["user"] = data["user_id"].map(user2user_encoded)
		data["movie"] = data["movie_id"].map(movie2movie_encoded)
		data["title_d"] = data["title"].map(title2title_encoded)
		self.searches["search_hist"] = self.searches["search_hist"].map(search_encoded)
		data["occupation"] = data["occupation"].map(occupations_encoded)
		self.searches = utils.normalize_col(self.searches,'search_hist')


		watch_hist = data.groupby(['user'])['movie_id'].apply(list).reset_index()
		search_hist = self.searches.groupby(['user'])['search_hist'].apply(list).reset_index()
		watch_hist_time = data.groupby(['user'])['watch_hist_time'].apply(list).reset_index()
		example_age = data.groupby(['user'])['example_age'].apply(list).reset_index()

		user_video_list = data.pivot(index='user_id', columns='movie_id', values='movie').reset_index()
		user_video_list.fillna(data["movie_id"].max()+1, inplace=True)

		sample_data=data[['user','occupation','sex']]
		sample_data=sample_data.reset_index()
		sample_data = sample_data.drop('index',axis=1)
		sample_data = sample_data.drop_duplicates()

		user_movie_list = pd.merge(sample_data,watch_hist, how= 'left')
		user_movie_list = pd.merge(user_movie_list,watch_hist_time, how='left')
		user_movie_list = pd.merge(user_movie_list,search_hist, how='left')
		user_movie_list = pd.merge(user_movie_list,example_age, how='left')
		user_movie_list['search_hist'] = user_movie_list['search_hist'].apply(lambda x: x if type(x) is list else []) # NaN 처리
		user_movie_list['predict_labels'] = user_movie_list['movie_id'].apply(lambda x: int(random.uniform(0,data["movie"].max()))) #label을 마지막 값으로..


		if(dataset_type == "train"):
			self.data = user_movie_list[(user_movie_list.user >= 1)&
		                                  (user_movie_list.user <= 5)]
		else:
			self.data = user_movie_list[(user_movie_list.user >= 6)&
		                                  (user_movie_list.user <= 10)]


	def preprocess_ranking(self, k):
		movie_data = self.movies.set_index(['movie_id']).sort_index()
		movie_data = movie_data.loc[k+1]
		movie_data["title_d"] = movie_data["title"].map(title2title_encoded)

		ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
		ratings = pd.read_csv(
		'data/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

		utils.get_genres(movie_data, self.genre_cols)

		new_data = movie_data.merge(ratings, on='movie_id') # rating 추가

		genre_occurences = new_data[genre_cols].sum().to_dict()
		genres_encoded = {x: i for i, x in enumerate(self.genre_cols)}


		new_data = new_data[['movie_id', 'user_id', 'rating', 'unix_timestamp', 'all_genres', 'title_d']]
		new_data['movie_type'] = np.where(new_data['rating'] >= 3, 'like', 'dislike') # 3보다 크면 like


		genre_list = new_data.groupby(['user_id'])['all_genres'].unique().apply(list).reset_index()
		genre_list['all_genres']=genre_list['all_genres'].apply(lambda x: list(set(','.join(x))) ) # 중복제거
		genre_list['all_genres']=genre_list['all_genres'].apply(lambda x:[ x for x in x if x.isdigit() ])

		new_data = normalize_col(new_data, 'unix_timestamp')
		timestamp_list = new_data.groupby(['user_id'])['unix_timestamp'].unique().apply(list).reset_index()

		title_list = new_data.groupby(['user_id'])['title_d'].apply(list).reset_index()
		print(title_list)
		dataset = movie_list.pivot(index='user_id', columns='movie_type', values='movie_id').reset_index()
		dataset.fillna(new_data["movie_id"].max()+1, inplace=True)

		dataset['like'] =dataset['like'].apply(lambda x: x if type(x) is list else [])
		dataset['dislike'] =dataset['dislike'].apply(lambda x: x if type(x) is list else [])

		dataset = pd.merge(dataset, title_list, how='left')
		dataset = pd.merge(dataset, genre_list, how='left')
		dataset = pd.merge(dataset, timestamp_list, how='left')

		dataset['predict_labels'] = dataset['like'].apply(lambda x: int(random.uniform(1,new_data["movie_id"].max()))) #label을 마지막 값으로..

		dataset['like']=dataset['like'].apply(lambda x: [new_data["movie_id"].max()+1] if x == [] else x)
		dataset['dislike']=dataset['dislike'].apply(lambda x: [new_data["movie_id"].max()+1] if x == [] else x)

		return dataset[(dataset.user_id >= 1)&
		                      (dataset.user_id <= 5)]
		# test_data=dataset[(dataset.user_id >= 6)&
		#                       (dataset.user_id <= 9)]

