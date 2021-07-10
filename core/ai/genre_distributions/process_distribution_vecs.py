import pickle
import pandas as pd
from codecs import encode, decode
import numpy as np
import time

# BEFORE THIS SCRIPT YOU MUST CREATE user_distrib_dump.csv. DO THIS BY RUNNING THE FOLLOWING FROM PSQL:
# \copy (SELECT user_id, encode(user_data, 'hex') FROM genre_distributions) to 'user_distrib_dump.csv'

start = time.time()
data = pd.read_csv('user_distrib_dump.csv', delimiter='\t', header=None, names=['user_id', 'user_data'])
loaded = time.time()
print(f"Read csv in {loaded - start}s")

stacked = np.vstack([pickle.loads(decode(x, 'hex')) for x in data['user_data']])
stacked_time = time.time()
print(f"Stacked in {stacked_time - loaded}s")

np.savetxt('user_distrib_dump.gz', stacked)
print(f"Saved in {time.time() - stacked_time}s")
