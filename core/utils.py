from urllib.request import urlretrieve
import zipfile
import pandas as pd
import numpy as np
import random

def get_genres(movies, genres):
	def get_all_genres(gs):
		active = [str(genres_encoded[genre]) for genre, g in zip(genres, gs) if g==1]
		if len(active) == 0:
			return '0'
		return ','.join((active))
	
	movies['all_genres'] = [
	get_all_genres(gs) for gs in zip(*[movies[genre] for genre in genres])]


def normalize_col(df,col_name):
	df[col_name] = (df[col_name] - df[col_name].min()) / (df[col_name].max() - df[col_name].min())
	return df



	