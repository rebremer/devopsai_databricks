import os,sys
import urllib.request
from .utils import load_data

class LoadData:

    def download_data(self):
        os.makedirs('./data', exist_ok = True)
        urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', filename='./data/train-images.gz')
        urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', filename='./data/train-labels.gz')
        urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename='./data/test-images.gz')
        urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', filename='./data/test-labels.gz')

    def train_test_split(self):
        x_train = load_data('./data/train-images.gz', False) / 255.0
        y_train = load_data('./data/train-labels.gz', True).reshape(-1)
        x_test = load_data('./data/test-images.gz', False) / 255.0
        y_test = load_data('./data/test-labels.gz', True).reshape(-1)
        return x_train,y_train,x_test,y_test

    def load_data_to_blob(self,ws):
        ds = ws.get_default_datastore()
        print(ds.datastore_type, ds.account_name, ds.container_name)
        ds.upload(src_dir='./data', target_path='mnist', overwrite=True, show_progress=True)    