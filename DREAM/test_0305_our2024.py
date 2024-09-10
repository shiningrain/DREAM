# run in autokeras_pure env
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('./utils')
from load_test_utils import traversalDir_FirstDir
from utils_data import *
from keras.datasets import mnist,cifar10,fashion_mnist,cifar100
from keras.models import load_model
from keras import backend as K
import pandas
import keras
import sys
import pickle
import tensorflow as tf
import time
import argparse
import shutil

IMAGE_SIZE=300# food101
#360×240 # cars196

def find_origin(root_dir):
    dir_list=traversalDir_FirstDir(root_dir)
    for trial_dir in dir_list:
        if os.path.basename(trial_dir).startswith('0-'):
            return trial_dir

def format_example(image, label,image_size=IMAGE_SIZE):
    # image_size=256
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, (image_size, image_size))#(360,240))#
    return image, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test image classification')
    # parser.add_argument('--method','-m',default='auto', help='model path')# 'auto' 'cust'
    parser.add_argument('--data','-d',default='cifar100',choices=['cifar10','cifar100','fashion','mnist','svhn','stl','emnist','food','cars','tiny'], help='dataset')
    parser.add_argument('--type','-t',default='ak_test',choices=['ak','ak_test'], help='model path')
    parser.add_argument('--trial_num_path','-tnp',default='/home/zxy/main/DL_autokeras/test_codes/1_evaluation/num.pkl',help='num pkl path')
    parser.add_argument('--epoch','-ep',default=10, help='training epoch')
    parser.add_argument('--tuner','-tn',default='greedy', help='training epoch')
    parser.add_argument('--trials','-tr',default=15, help='searching trials')
    # parser.add_argument('--root_save_path','-rsp',default='/home/zxy/main/DL_autokeras/test_codes/1_evaluation/tiny_test_2024_root_save_path.pkl', help='orgin autokeras path') 
    parser.add_argument('--root_path','-rp',default='/home/zxy/main/DL_autokeras/test_codes/1_evaluation/c100_test_2024', help='orgin autokeras path') 
    parser.add_argument('--origin_param_path','-opp',default='/home/zxy/main/DL_autokeras/test_codes/opensource/ExperimentRQ1/intial_param/param_c100_1.pkl', help='orgin autokeras path')
    parser.add_argument('--tmp_dir','-td',default='./Test_dir/tmp0', help='the directory to save results')    
    args = parser.parse_args()


    import autokeras as ak
    root_path=args.root_path
    tmp_dir=args.tmp_dir#os.path.join(os.path.dirname(os.path.abspath(root_path)),'tmp')
    if os.path.exists(root_path):
        # os._exit(0)
        shutil.rmtree(root_path)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(root_path)
    os.makedirs(tmp_dir)

    log_path=os.path.join(root_path,'log.pkl')

    if args.data=='mnist':
        (x_train, y_train), (x_test, y_test)=mnist_load_data()
    # elif args.data=='cifar10':
    #     (x_train, y_train), (x_test, y_test)=cifar10_load_data()
    elif args.data=='cifar100':
        (x_train, y_train), (x_test, y_test)=cifar100_load_data()
    elif args.data=='fashion':
        (x_train, y_train), (x_test, y_test)=fashion_load_data()
    elif args.data=='svhn':
        (x_train, y_train), (x_test, y_test)=svhn_load_data()
    elif args.data=='stl':
        (x_train, y_train), (x_test, y_test)=stl_load_data()
    elif args.data=='emnist':
        (x_train, y_train), (x_test, y_test)=emnist_load_data() 
    # for i in range(5):
    start_time=time.time()
    


    if not os.path.exists(log_path):
        log_dict={}
        log_dict['cur_trial']=-1
        log_dict['start_time']=time.time()
        log_dict['param_path']=os.path.abspath(args.origin_param_path)
        log_dict['data']=args.data
        log_dict['tmp_dir']=args.tmp_dir

        with open(log_path, 'wb') as f:
            pickle.dump(log_dict, f)
    else:
        with open(log_path, 'rb') as f:#input,bug type,params
            log_dict = pickle.load(f)
            # log_dict['cur_trial']+=1
        for key in log_dict.keys():
            if key.startswith('{}-'.format(log_dict['cur_trial'])):
                log_dict['start_time']=time.time()-log_dict[key]['time']
                break
        with open(log_path, 'wb') as f:
            pickle.dump(log_dict, f)


    # if args.method=='auto':
    clf = ak.ImageClassifier(
    overwrite=True,directory=os.path.join(args.root_path,'image_classifier'),
    max_trials=int(args.trials),tuner=args.tuner)#,tuner='bayesian'

    if args.epoch!=None:
        epoch=int(args.epoch)
    else:
        epoch=args.epoch

    import tensorflow_datasets as tfds
    # IMAGE_SIZE=args.image_size
    if args.data=='food':
        train = tfds.load('food101',data_dir='/home/zxy/main/DL_autokeras/test_codes/1_dataset',
                split='train',as_supervised=True)# shuffle_files=True,  ,batch_size=32
        train = train.map(format_example)
        clf.fit(train, epochs=epoch,root_path=root_path)
    elif  args.data=='cars':
        train = tfds.load('cars196',data_dir='/home/zxy/main/DL_autokeras/test_codes/1_dataset/cars196',
                split='train',as_supervised=True)# shuffle_files=True, ,batch_size=32
        train = train.map(format_example)
        clf.fit(train, epochs=epoch,root_path=root_path)
    elif args.data=='tiny':
        from tiny_imagenet import TinyImagenetDataset
        tf.compat.v1.enable_eager_execution()
        tiny_imagenet_builder = TinyImagenetDataset(data_dir='/home/zxy/main/DL_autokeras/test_codes/1_dataset')
        train = tiny_imagenet_builder.as_dataset(split="train",as_supervised=True)
        # train = tfds.load('tiny_imagenet_dataset',data_dir='/home/zxy/workspace/DL_work/DL_autokeras/1Autokeras/test_codes/experiment/1_dataset',
        #     split='train',as_supervised=True)# shuffle_files=True, 
        # tf.compat.v1.data.get_output_types(train)
        clf.fit(train, epochs=epoch,root_path=root_path)
    elif args.data=='cifar10':
        from tiny_imagenet import TinyImagenetDataset
        tf.compat.v1.enable_eager_execution()
        tiny_imagenet_builder = TinyImagenetDataset(data_dir='/home/zxy/main/DL_autokeras/test_codes/1_dataset')
        train = tiny_imagenet_builder.as_dataset(split="train",as_supervised=True)
        # train = tfds.load('tiny_imagenet_dataset',data_dir='/home/zxy/workspace/DL_work/DL_autokeras/1Autokeras/test_codes/experiment/1_dataset',
        #     split='train',as_supervised=True)# shuffle_files=True, 
        # tf.compat.v1.data.get_output_types(train)
        clf.fit(train, epochs=epoch,root_path=root_path)
    else:
        clf.fit(x_train, y_train, epochs=epoch,root_path=root_path)

    print('finish')
