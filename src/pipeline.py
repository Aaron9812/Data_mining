from src.data.dataset import download_dataset, import_data
import pandas as pd
from tqdm import tqdm
from src.data.preprocessing import preprocess
import logging
import os
import time

tqdm.pandas()
LOGGER = logging.getLogger('pipeline')


class Pipeline(object):
    """ Class to combine the different download, preprocessing, modeling and evaluation steps. """

    tweets = None

    def __init__(self, tweets: str = None):
        if tweets is not None:
            self.tweets = pd.read_pickle(tweets)

    def setup(self, datasets: list = None, path: str = 'data/tweets')
	
		# ...
        
		return self.save()

    def preprocess(self):
	
		# ...
		
        return self.save()

    def save(self, path: str = 'data/processed'):
	
		# ...
		
        return self
