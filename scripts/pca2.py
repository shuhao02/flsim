import argparse
import client
import config
import logging
import os
import pickle
from sklearn.decomposition import PCA


def pca(weights):
    def flatten_weights(weights):
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten())
        return weight_vecs

    logging.info('Flattening weights...')
    weight_vecs = [flatten_weights(weight) for weight in weights]

    pca = PCA(n_components=2)
    new_state = pca.fit_transform(weight_vecs)

    return new_state