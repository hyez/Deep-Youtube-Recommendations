import tensorflow as tf
import datetime
import sys, os
import numpy as np
sys.path.append(os.pardir)
import core.utils as utils
import core.config as cfg
from model.layers import MaskedEmbeddingsAggregatorLayer, L2NormLayer
from model.candidate_generation import CandidateGeneration
from model.ranking import Ranking
# from core.dataset import Dataset

class DeepRecommendations(object):
	def __init__(self, input_data):
		self.trainset = input_data
		pred = CandidateGeneration().build_nework().predict([tf.keras.preprocessing.sequence.pad_sequences(self.trainset['movie_id']),
           tf.keras.preprocessing.sequence.pad_sequences(self.trainset['watch_hist_time'], dtype=float),
           tf.keras.preprocessing.sequence.pad_sequences(self.trainset['search_hist'], dtype=float) + 1e-10,
           tf.keras.preprocessing.sequence.pad_sequences(self.trainset['example_age'], dtype=float),
           ])

		print(pred)
		# candidate generation: 
		###### 각 user당 top-7개의 추천 데이터를 뽑아낸다.
		# movies = utils.get_topk(6)
		N = 6
		k = np.sort((-pred).argsort()[:,:N])
		print(k)
		k = k.flatten()
		# k[k>data["movie"].max()]=0
		k = np.unique(k)
