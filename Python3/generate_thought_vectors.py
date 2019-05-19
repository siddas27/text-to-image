import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts
import h5py

def gen_thought(text):
	captions =[]
	captions.append(text)
	print(captions)
	model = skipthoughts.load_model()
	caption_vectors = skipthoughts.encode(model, captions)

	if os.path.isfile(join('Data', 'sample_caption_vectors.hdf5')):
		os.remove(join('Data', 'sample_caption_vectors.hdf5'))
	h = h5py.File(join('Data', 'sample_caption_vectors.hdf5'))
	h.create_dataset('vectors', data=caption_vectors)		
	h.close()
