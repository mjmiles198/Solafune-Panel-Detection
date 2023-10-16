import glob
import tifffile
import numpy as np
import lightgbm as lgb
import warnings
import tqdm

train_path =  '/tmp/kaggledata/solafune_solars/train/s2_image/'
mask_path = '/tmp/kaggledata/solafune_solars/train/mask/'

masks = glob.glob(f'{mask_path}/*')
trains = glob.glob(f'{train_path}/*')
masks.sort()
trains.sort()


X = []
y = []
g = []

for i, (t, m) in enumerate(zip(trains, masks)):
    img = tifffile.imread(t).astype(np.float)
    mask = tifffile.imread(m).astype(np.float)
    X.append(img.reshape(-1,12))
    y.append(mask.reshape(-1))
    g.append(np.ones_like(mask.reshape(-1))*i)
    
X = np.vstack(X)
y = np.hstack(y)
g = np.hstack(g) 