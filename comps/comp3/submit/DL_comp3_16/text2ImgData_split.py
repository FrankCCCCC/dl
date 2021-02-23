#%%
import pickle
import os
import time
import random
import pandas as pd
import numpy as np
from IPython.display import display
#%%
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_t2i(path):
    return load_obj(path)

def split_train(t2i, name=None):
    sig_embed2img = pd.DataFrame(columns=t2i.columns)
    # display(sig_embed2img)
    embs = []
    imgs = []
    for embed, img in zip(t2i['Captions'], t2i['ImagePath']):
        embs.extend(embed)
        imgs.extend([img[13:]] * len(embed))
        # print([img] * len(embed))

    for id, cap in enumerate(t2i['Captions'][:5]):
        print(f"{id}: {len(cap)}")

    sig_embed2img['Captions'] = embs
    sig_embed2img['ImagePath'] = np.array(imgs)

    print(name)
    display(sig_embed2img.head(50))
    print(sig_embed2img.shape)
    return sig_embed2img

def split_test(t2i, name=None):
    sig_embed2img = pd.DataFrame(columns=t2i.columns)
    embs = []
    for embed in t2i['Captions']:
        embs.extend(embed)

    for id, cap in enumerate(t2i['Captions'][:5]):
        print(f"{id}: {len(cap)}")

    sig_embed2img['Captions'] = embs
    # sig_embed2img['ImagePath'] = np.array(imgs)

    print(name)
    display(sig_embed2img.head(50))
    print(sig_embed2img.shape)
    return sig_embed2img

# %%
# t2i_path = '/home/weidagogo/chi-shen/comp3/DCGAN_ours/datalab-cup3-reverse-image-caption-2020/dataset/dataset/text2ImgData'
t2i_train_path = "/home/weidagogo/chi-shen/comp3/DCGAN_ours/datalab-cup3-reverse-image-caption-2020/dataset/dataset/embed_image/train"
t2i_test_path = "/home/weidagogo/chi-shen/comp3/DCGAN_ours/datalab-cup3-reverse-image-caption-2020/dataset/dataset/embed_image/test"
t2i_train = load_t2i(t2i_train_path)
t2i_test = load_t2i(t2i_test_path)

print("t2i_train")
display(t2i_train)
print(t2i_train.shape)

print("t2i_test")
display(t2i_test)
print(t2i_test.shape)
# %%
sig_embed2img_train = split_train(t2i_train, 't2i_train')
sig_embed2img_test = split_test(t2i_test, 't2i_test')
# %%
save_obj(sig_embed2img_train, "/home/weidagogo/chi-shen/comp3/DCGAN_ours/datalab-cup3-reverse-image-caption-2020/dataset/dataset/embed_image/train_split")
save_obj(sig_embed2img_test, "/home/weidagogo/chi-shen/comp3/DCGAN_ours/datalab-cup3-reverse-image-caption-2020/dataset/dataset/embed_image/test_split")
# %%
