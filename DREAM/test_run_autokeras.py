# run in autokeras_pure env
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('./utils')
from load_test_utils import traversalDir_FirstDir
from utils_data import *
# from tensorflow.keras.datasets import mnist,cifar10,fashion_mnist,cifar100
# from tensorflow.keras.models import load_model
# from tensorflow.keras import backend as K
# import keras
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
    image = tf.image.resize(image, (image_size, image_size))
    return image, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test image classification')
    parser.add_argument('--data','-d',default='cifar100',choices=['cifar100','food','cars','tiny'], help='dataset')
    parser.add_argument('--trial_num_path','-tnp',default='/home/zxy/main/DL_autokeras/test_codes/DREAM/DREAM/Test_dir/autokeras_result/num.pkl',help='num pkl path')
    parser.add_argument('--epoch','-ep',default=200, help='training epoch')
    parser.add_argument('--tuner','-tn',default='greedy', help='training epoch')# hyperband, bayesian
    parser.add_argument('--trials','-tr',default=50, help='searching trials')
    parser.add_argument('--root_save_path','-rsp',default='/home/zxy/main/DL_autokeras/test_codes/DREAM/DREAM/Test_dir/autokeras_result/c100_test_root_save_path.pkl', help='orgin autokeras path') 
    parser.add_argument('--root_path','-rp',default='/home/zxy/main/DL_autokeras/test_codes/DREAM/DREAM/Test_dir/autokeras_result/c100_test', help='orgin autokeras path') 
    parser.add_argument('--origin_param_path','-op',default='/home/zxy/main/DL_autokeras/test_codes/DREAM/DREAM/Test_dir/demo_origin/param_c100_test.pkl', help='orgin autokeras path') 
    args = parser.parse_args()

    import autokeras as ak
    root_path=args.root_path
    log_path=os.path.join(args.root_path,'log.pkl')
    
    if os.path.exists(root_path):
        import shutil
        shutil.rmtree(root_path)
        

    if not os.path.exists(root_path):
        os.makedirs(root_path)
    else:
        os._exit(0)

    with open(args.root_save_path, 'wb') as f:
        save_dict={}
        save_dict['root_path']=root_path
        save_dict['log_path']=log_path
        save_dict['num_path']=args.trial_num_path
        save_dict['tuner']=args.tuner
        pickle.dump(save_dict, f)

    if args.data=='cifar100':
        (x_train, y_train), (x_test, y_test)=cifar100_load_data()

    start_time=time.time()


        
    if not os.path.exists(log_path):
        log_dict={}
        log_dict['cur_trial']=-1
        log_dict['start_time']=time.time()
        log_dict['origin_param_path']=args.origin_param_path

        with open(log_path, 'wb') as f:
            pickle.dump(log_dict, f)

        # delete dir!


    # if args.method=='auto':
    clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=int(args.trials),tuner=args.tuner)#,tuner='bayesian'

    if args.epoch!=None:
        epoch=int(args.epoch)
    else:
        epoch=args.epoch
    
    import tensorflow_datasets as tfds

    if args.data=='food':
        train = tfds.load('food101',data_dir='your_dataset_path',
                split='train',as_supervised=True)
        train = train.map(format_example)
        clf.fit(train, epochs=epoch,root_path=args.root_save_path)
    elif  args.data=='cars':
        train = tfds.load('cars196',data_dir='your_dataset_path',
                split='train',as_supervised=True)
        train = train.map(format_example)
        clf.fit(train, epochs=epoch,root_path=args.root_save_path)
    elif args.data=='tiny':
        from tiny_imagenet import TinyImagenetDataset
        tf.compat.v1.enable_eager_execution()
        tiny_imagenet_builder = TinyImagenetDataset(data_dir='your_dataset_path')
        train = tiny_imagenet_builder.as_dataset(split="train",as_supervised=True)
        clf.fit(train, epochs=epoch,root_path=args.root_save_path)
    else:
        clf.fit(x_train, y_train, epochs=epoch,root_path=args.root_save_path)


    print('finish')