#---inputs
import tensorflow as tf
import datetime
import sys, os
sys.path.append(os.pardir)

import core.config as cfg
from model.layers import MaskedEmbeddingsAggregatorLayer, L2NormLayer

class Ranking(object):
	def __init__(self, trainable=True):

		self.trainable = trainable
	
	def build_nework(self):

		input_title = tf.keras.Input(shape=(None, ), name='movie_name')
		inp_video_liked = tf.keras.layers.Input(shape=(None,), name='like')
		inp_video_disliked = tf.keras.layers.Input(shape=(None,), name='dislike')
		input_genre = tf.keras.Input(shape=(None, ), name='genre')
		input_timestamp = tf.keras.Input(shape=(None, ), name='timestamp')


		#--- layers
		features_embedding_layer = tf.keras.layers.Embedding(input_dim=cfg.NUM_CLASSES, output_dim=cfg.EMBEDDING_DIMS, 
		                                            mask_zero=True, trainable=True, name='features_embeddings')
		labels_embedding_layer = tf.keras.layers.Embedding(input_dim=cfg.NUM_CLASSES, output_dim=cfg.EMBEDDING_DIMS, 
		                                            mask_zero=True, trainable=True, name='labels_embeddings')

		avg_embeddings = MaskedEmbeddingsAggregatorLayer(agg_mode='mean', name='aggregate_embeddings')

		dense_1 = tf.keras.layers.Dense(units=cfg.DENSE_UNITS, name='dense_1')
		dense_2 = tf.keras.layers.Dense(units=cfg.DENSE_UNITS, name='dense_2')
		dense_3 = tf.keras.layers.Dense(units=cfg.DENSE_UNITS, name='dense_3')
		l2_norm_1 = L2NormLayer(name='l2_norm_1')

		dense_output = tf.keras.layers.Dense(cfg.NUM_CLASSES, activation=tf.nn.softmax, name='dense_output')

		#--- features
		features_embeddings = features_embedding_layer(input_title)
		l2_norm_features = l2_norm_1(features_embeddings)
		avg_features = avg_embeddings(l2_norm_features)

		labels_liked_embeddings = labels_embedding_layer(inp_video_liked)
		l2_norm_liked = l2_norm_1(labels_liked_embeddings)
		avg_liked = avg_embeddings(l2_norm_liked)

		labels_disliked_embeddings = labels_embedding_layer(inp_video_disliked)
		l2_norm_disliked = l2_norm_1(labels_disliked_embeddings)
		avg_disliked = avg_embeddings(l2_norm_disliked)

		labels_genre_embeddings = labels_embedding_layer(input_genre)
		l2_norm_genre = l2_norm_1(labels_genre_embeddings)
		avg_genre = avg_embeddings(l2_norm_genre)

		labels_timestamp_embeddings = labels_embedding_layer(input_timestamp)
		l2_norm_timestamp = l2_norm_1(labels_timestamp_embeddings)
		avg_timestamp = avg_embeddings(l2_norm_timestamp)


		# 임베딩 벡터들 연결
		concat_inputs = tf.keras.layers.Concatenate(axis=1)([avg_features,
		                                                     avg_liked,
		                                                     avg_disliked,
		                                                     avg_genre,
		                                                     avg_timestamp
		                                                     ])
		# Dense Layers
		dense_1_features = dense_1(concat_inputs)
		dense_1_relu = tf.keras.layers.ReLU(name='dense_1_relu')(dense_1_features)
		dense_1_batch_norm = tf.keras.layers.BatchNormalization(name='dense_1_batch_norm')(dense_1_relu)

		dense_2_features = dense_2(dense_1_relu)
		dense_2_relu = tf.keras.layers.ReLU(name='dense_2_relu')(dense_2_features)
		# dense_2_batch_norm = tf.keras.layers.BatchNormalization(name='dense_2_batch_norm')(dense_2_relu)

		dense_3_features = dense_3(dense_2_relu)
		dense_3_relu = tf.keras.layers.ReLU(name='dense_3_relu')(dense_3_features)
		dense_3_batch_norm = tf.keras.layers.BatchNormalization(name='dense_3_batch_norm')(dense_3_relu)
		outputs = dense_output(dense_3_batch_norm)

		#Optimizer
		optimiser = tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)

		#--- prep model
		model = tf.keras.models.Model(
		    inputs=[input_title, 
		            inp_video_liked, 
		            inp_video_disliked,
		            input_genre,
		            input_timestamp,
		            ],
		    outputs=[outputs]
		)
		logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
		tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
		model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['acc'])

		self.model = model

		return model

	def save(self):
		self.model.save("ranking.h5")

	def summary(self):
		self.model.summary()
	