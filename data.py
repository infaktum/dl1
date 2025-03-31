import urllib.request
from pathlib import Path
from PIL import Image
import pandas as pd


data_dir = './daten'

def load_image(fname, url = None):
    fname = f'{data_dir}/{fname}'
    if not Path(fname).exists() and url:    
        
        urllib.request.urlretrieve(url,fname)
    return Image.open(fname)        

from sklearn.datasets import fetch_openml

def load_mnist():
    if not Path(f'{data_dir}/mnist_data.csv').exists():
        mnist = fetch_openml('mnist_784', version=1, parser = "auto")
        mnist.data.to_csv(f'{data_dir}/mnist_data.csv',index = False)
        mnist.target.to_csv(f'{data_dir}/mnist_target.csv',index=False)
        return mnist.data, mnist.target.astype(int)
    else:
        return pd.read_csv(f'{data_dir}/mnist_data.csv'), pd.read_csv(f'{data_dir}/mnist_target.csv').astype(int)

def load_imagenet_class_labels():
    path = f'{data_dir}/imagenet_class_labels.txt'
    
    if not Path(path).exists():
        url = 'https://raw.githubusercontent.com/joe-papa/pytorch-book/main/files/imagenet_class_labels.txt'        
        urllib.request.urlretrieve(url,path)
    with open(path) as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

espresso = load_image('coffee.jpg', 'https://upload.wikimedia.org/wikipedia/commons/4/45/A_small_cup_of_coffee.JPG')
leuchtturm = load_image('Leuchtturm.jpg')

def main():
    load_mnist()
    load_imagenet_class_labels()


if __name__ == '__main__':
    main()