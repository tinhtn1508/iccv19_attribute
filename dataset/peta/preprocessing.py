import os
import numpy as np
import random
import pickle
from scipy.io import loadmat
from easydict import EasyDict
import math

np.random.seed(0)
random.seed(0)

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    """
    dataset = dict()
    dataset['description'] = 'peta'
    dataset['root'] = './images/'
    dataset['image'] = []
    dataset['att'] = []
    dataset['att_name'] = []
    dataset['selected_attribute'] = range(35)
    # load PETA.MAT
    data = loadmat(open('./PETA.mat', 'rb'))

    for idx in range(105):
        dataset['att_name'].append(data['peta'][0][0][1][idx,0][0])

    for idx in range(19000):
        dataset['image'].append('%05d.png'%(idx+1))
        dataset['att'].append(data['peta'][0][0][0][idx, 4:].tolist())

    with open(os.path.join(save_dir, 'peta_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

def create_trainvaltest_split(traintest_split_file):
    """
    create a dataset split file, which consists of index of the train/val/test splits
    """
    dataset = EasyDict()
    dataset.description = 'pa100k'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, '/home/tinhtn/workspace/iccv19_attribute/dataset/peta/images')


    # load PETA.MAT
    data = loadmat(open('./PETA.mat', 'rb'))
    dataset.attr_name = [data['peta'][0][0][1][idx,0][0] for idx in range(105)]
    dataset.label = np.array([data['peta'][0][0][0][idx, 4:].tolist() for idx in range(19000)])
    print(dataset.label.shape)

    train_image_name_int = []
    test_image_name_int = []

    for idx in range(5):
        train = (data['peta'][0][0][3][idx][0][0][0][0][:,0]-1).tolist()
        val = (data['peta'][0][0][3][idx][0][0][0][1][:,0]-1).tolist()
        test = (data['peta'][0][0][3][idx][0][0][0][2][:,0]-1).tolist()
        trainval = train + val
        train_image_name_int.append(trainval)
        test_image_name_int.append(test)

    train_image_name = ['%05d.png'%(i+1) for i in train_image_name_int[0]]
    test_image_name = ['%05d.png'%(i+1) for i in test_image_name_int[0]]

    dataset.image_name = train_image_name + test_image_name
    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 11400)  # np.array(range(80000))
    # dataset.partition.val = np.arange(90000, 100000)  # np.array(range(80000, 90000))
    dataset.partition.test = np.arange(11400, 19000)  # np.array(range(90000, 100000))
    with open(traintest_split_file, 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="peta dataset")
    parser.add_argument(
        '--save_dir',
        type=str,
        default='.')
    parser.add_argument(
        '--traintest_split_file',
        type=str,
        default="./peta_partition.pkl")
    args = parser.parse_args()
    save_dir = args.save_dir
    traintest_split_file = args.traintest_split_file

    # generate_data_description(save_dir)
    create_trainvaltest_split(traintest_split_file)
