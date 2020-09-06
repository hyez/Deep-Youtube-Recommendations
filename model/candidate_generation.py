#---inputs
import tensorflow as tf
import datetime
import sys, os
sys.path.append(os.pardir)

import core.config as cfg
from model.layers import MaskedEmbeddingsAggregatorLayer, L2NormLayer

class CandidateGeneration(object):
    def __init__(self, trainable=True):

        self.trainable = trainable

    def build_nework(self):
        input_watch_hist = tf.keras.Input(shape=(None, ), name='watch_hist')
        input_watch_hist_time = tf.keras.layers.Input(shape=(None,), name='watch_hist_time')
        input_search_hist = tf.keras.layers.Input(shape=(None,), name='search_hist')
        input_example_age = tf.keras.Input(shape=(None, ), name='example_age')

        #--- layers
        features_embedding_layer = tf.keras.layers.Embedding(input_dim=cfg.NUM_CLASSES, output_dim=cfg.EMBEDDING_DIMS, mask_zero=True, trainable=True, name='features_embeddings')
        labels_embedding_layer = tf.keras.layers.Embedding(input_dim=cfg.NUM_CLASSES, output_dim=cfg.EMBEDDING_DIMS,mask_zero=True, trainable=True, name='labels_embeddings')

        avg_embeddings = MaskedEmbeddingsAggregatorLayer(agg_mode='mean', name='aggregate_embeddings')

        dense_1 = tf.keras.layers.Dense(units=cfg.DENSE_UNITS, name='dense_1')
        dense_2 = tf.keras.layers.Dense(units=cfg.DENSE_UNITS, name='dense_2')
        dense_3 = tf.keras.layers.Dense(units=cfg.DENSE_UNITS, name='dense_3')
        l2_norm_1 = L2NormLayer(name='l2_norm_1')

        dense_output = tf.keras.layers.Dense(cfg.NUM_CLASSES, activation=tf.nn.softmax, name='dense_output')

        #--- features
        features_embeddings = features_embedding_layer(input_watch_hist)
        l2_norm_features = l2_norm_1(features_embeddings)
        avg_features = avg_embeddings(l2_norm_features)

        labels_watch_embeddings = labels_embedding_layer(input_watch_hist_time)
        l2_norm_watched = l2_norm_1(labels_watch_embeddings)
        avg_watched = avg_embeddings(l2_norm_watched)

        labels_search_embeddings = labels_embedding_layer(input_search_hist)
        l2_norm_searched = l2_norm_1(labels_search_embeddings)
        avg_searched = avg_embeddings(l2_norm_searched)

        labels_example_age_embeddings = labels_embedding_layer(input_example_age)
        l2_norm_example_age = l2_norm_1(labels_example_age_embeddings)
        avg_example_age = avg_embeddings(l2_norm_example_age)


        # 임베딩 벡터들 연결
        concat_inputs = tf.keras.layers.Concatenate(axis=1)([avg_features,
                                                             avg_watched,
                                                             avg_searched,
                                                             avg_example_age,
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
            inputs=[input_watch_hist, 
                    input_watch_hist_time, 
                    input_search_hist,
                    input_example_age,
                    ],
            outputs=[outputs]
        )
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['acc'])

        self.model = model

        return model

    def save(self):
        self.model.save("candidate_generation.h5")

    def summary(self):
        self.model.summary()
