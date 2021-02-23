#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
#%%
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
# %%
path_train_cap = "/home/weidagogo/chi-shen/comp3/DCGAN_ours/datalab-cup3-reverse-image-caption-2020/dataset/dataset/embed_image/train"
train_cap = load_obj(path_train_cap)
display(train_cap)
print("Embedding Size of first sentence of first image")
print(len(train_cap['Captions'][1][0]))
# %%
path_test_cap = "/home/weidagogo/chi-shen/comp3/DCGAN_ours/datalab-cup3-reverse-image-caption-2020/dataset/dataset/embed_image/test"
test_cap = load_obj(path_test_cap)
display(test_cap)
# %%
