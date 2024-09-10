import copy
import os

import kerastuner
import tensorflow as tf
from kerastuner.engine import hypermodel as hm_module
from tensorflow.keras import callbacks as tf_callbacks
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.util import nest

from autokeras import pipeline as pipeline_module
from autokeras.utils import data_utils
from autokeras.utils import utils

import tensorflow.keras as keras
import pickle
import numpy as np
import nni

def read_data(dataset,batch_size):
    # read data from a new unzipped dataset.
    trainX=dataset['x'][:batch_size]
    trainy=dataset['y'][:batch_size]
    testX=dataset['x_val'][:batch_size]
    testy=dataset['y_val'][:batch_size]
    return trainX,trainy,testX,testy


class LossHistory(keras.callbacks.Callback):

    def __init__(self,training_data,model,total_epoch,batch_size,save_path): #only support epoch method now
        """[summary]

        Args:
            training_data ([list]): [training dataset]
            model ([model]): [untrained model]
            batch_size ([int]): [batch size]
            save-dir([str]):[the dir to save the detect result]
            checktype (str, optional): [checktype,'a_b', a can be chosen from ['epoch', 'batch'], b is number, it means the monitor will check \
            the gradient and loss every 'b' 'a'.]. Defaults to 'epoch_5'.
            satisfied_acc (float, optional): [the satisfied accuracy, when val accuracy beyond this, the count ++, when the count is bigger or\
                equal to satisfied_count, training stop.]. Defaults to 0.7.

        """
        self.trainX,self.trainy,self.testX,self.testy = read_data(training_data,batch_size)
        self.model=model
        self.epoch=total_epoch
        self.save_path=os.path.abspath(save_path)
        self.tmp_dir=os.path.dirname(self.save_path)
        save_dict={}
        save_dict['gradient']={}
        save_dict['weight']={}
        with open(self.save_path, 'wb') as f:
            pickle.dump(save_dict, f)

        self.x_path=os.path.join(os.path.abspath(self.tmp_dir),'x.npy')
        self.y_path=os.path.join(os.path.abspath(self.tmp_dir),'y.npy')
        if 'text' in self.tmp_dir:
            self.model_path=os.path.join(os.path.abspath(self.tmp_dir),'model.h5py')
        else:
            self.model_path=os.path.join(os.path.abspath(self.tmp_dir),'model.h5')
        trainingExample = self.trainX
        trainingY=self.trainy
        np.save(self.x_path,trainingExample)
        np.save(self.y_path,trainingY)


    def on_epoch_end(self,epoch,logs={}):
        """Reports intermediate accuracy to NNI framework"""
        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])

        if self.model_path.endswith('h5py'):
            self.model.save(self.model_path,save_format='tf')
        else:
            try:
                self.model.save(self.model_path)
            except:
                import time
                time.sleep(5)
                os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
                self.model.save(self.model_path)
        get_gradient(self.model_path,self.x_path,self.y_path,epoch,self.save_path)


def get_gradient(model_path,x_path,y_path,epoch,save_path):
    import subprocess
    command="/home/zxy/main/anaconda3/envs/tf_ak_test/bin/python ../utils/get_gradient_on_cpu.py -m {} -dx {} -dy {} -ep {} -sp {}" #TODO:need to set your your python interpreter path

    out_path=save_path.split('.')[0]+'_out'
    out_file = open(out_path, 'w')
    out_file.write('logs\n')
    run_cmd=command.format(model_path,x_path,y_path,epoch,save_path)
    subprocess.Popen(run_cmd, shell=True, stdout=out_file, stderr=out_file)

def extract_dataset_np(data_x,tmp_dir,method='mnist'):
    tmp_path=os.path.join(tmp_dir,'{}.pkl'.format(method))
    if os.path.exists(tmp_path):
        with open(tmp_path, 'rb') as f:#input,bug type,params
            dataset = pickle.load(f)
        batch_size=dataset['batch']
        del dataset['batch']
    else:
        dataset={}
        dataset['x']=[]
        dataset['y']=[]
        dataset['x_val']=[]
        dataset['y_val']=[]
        batch_size=data_x[0].shape[0]
        batch1 = data_x[0]
        dataset['x']=data_x[0]
        dataset['y']=data_x[1]         
        dataset['batch']=batch_size
        with open(tmp_path, 'wb') as f:
            pickle.dump(dataset, f)
        del dataset['batch']
    return dataset,batch_size
