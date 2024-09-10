'''
Author: your name
Date: 2021-07-14 10:31:22
LastEditTime: 2021-08-31 21:09:23
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /test_codes/utils/operation_test_utils.py
'''
import pickle
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import autokeras as ak

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from tensorflow.keras.datasets import mnist,cifar10
import time
import gc
import numpy as np
import copy
import uuid

import kerastuner
import tensorflow as tf
from kerastuner.engine import hypermodel as hm_module
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.util import nest
# from autokeras.engine import compute_gradient as cg

def evaluate_training(train_history,method='val_accuracy'):
    '''
    input:
    train history
    train histroy is a dict includes: loss, val_loss, accuracy, val_accuracy from train history
    default evaluation method is using the bigges val_accuracy as the score (using in autokeras eval trials)

    return: 
    score [float]
    '''
    if method=='val_accuracy':
        try:
            score=max(train_history['val_acc'])
        except:
            score=max(train_history['val_accuracy'])
    return score

class LossHistory(keras.callbacks.Callback):

    def __init__(self,training_data,model,total_epoch,batch_size,save_path,tmp_save_dir): #only support epoch method now
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
        self.save_path=save_path
        save_dict={}
        save_dict['gradient']={}
        save_dict['weight']={}
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)

        self.x_path=os.path.join(tmp_save_dir,'x.npy')
        self.y_path=os.path.join(tmp_save_dir,'y.npy')
        self.model_path=os.path.join(tmp_save_dir,'model.h5py')
        trainingExample = self.trainX
        trainingY=self.trainy
        np.save(self.x_path,trainingExample)
        np.save(self.y_path,trainingY)

    def on_epoch_end(self,epoch,logs={}):
        # if (epoch)%self.checkgap==0:
            
        self.model.save(self.model_path)
        get_gradient(self.model_path,self.x_path,self.y_path,epoch,self.save_path)



def get_gradient(model_path,x_path,y_path,epoch,save_path):
    import subprocess
    command="/home/zxy/main/anaconda3/envs/tf_ak_test/bin/python ./utils/get_gradient_on_cpu.py -m {} -dx {} -dy {} -ep {} -sp {}" #TODO:need to set your your python interpreter path

    out_path=save_path.split('.')[0]+'_out'
    out_file = open(out_path, 'w')
    out_file.write('logs\n')
    run_cmd=command.format(model_path,x_path,y_path,epoch,save_path)
    subprocess.Popen(run_cmd, shell=True, stdout=out_file, stderr=out_file)


def read_data(dataset,batch_size):
    # read data from a new unzipped dataset.
    trainX=dataset['x'][:batch_size]
    trainy=dataset['y'][:batch_size]
    testX=dataset['x_val'][:batch_size]
    testy=dataset['y_val'][:batch_size]
    return trainX,trainy,testX,testy

def train_model(model,config,callbacks,save_dir):

    history = model.fit(config['dataset']['x'], config['dataset']['y'],batch_size=config['batch_size'], validation_data=(config['dataset']['x_val'], config['dataset']['y_val']),\
        epochs=config['epoch'],callbacks=callbacks)
    score=evaluate_training(history.history)
    history_path=os.path.join(save_dir,'history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    return history,score

def get_new_values(hyperparameter_space,current_value):
    if hasattr(hyperparameter_space,'values'):
        out_list=copy.deepcopy(hyperparameter_space.values)
        out_list.remove(current_value)
    else:
        if current_value:
            out_list=[False]
        if not current_value:
            out_list=[True]
    return out_list

def mutate_hp(hyperparameters,key):
    for hp in hyperparameters.space:
        if hp.name!=key:
            continue
        new_value_list=get_new_values(hp,hyperparameters.values[key])
        return new_value_list

def check_extend_operation(hps,new_hps):
    extend_list=[]
    hps_key_list=list(hps.values.keys())
    try:
        for key in new_hps.values.keys():
            if key not in hps_key_list:
                extend_list.append(key)
    except:
        # for special situations
        return extend_list
            
    return extend_list

def get_new_hps(hyperparameters,key,value):
    from kerastuner.engine import hyperparameters as hp_module# /kerastuner/engine/oracle.py line 395
    # while 1:
    hps = hp_module.HyperParameters()
    # Generate a set of random values.
    extend_list=[]
    for hp in hyperparameters.space:
        hps.merge([hp])
        if hps.is_active(hp):  # Only active params in `values`.
            # test for augmentations
            if hp.name in hyperparameters.values.keys():
                if hp.name==key:
                    hps.values[hp.name]=value
                else:
                    hps.values[hp.name] = hyperparameters.values[hp.name]
            else:
                hps.values[hp.name] = hp.random_sample(time.time())
                extend_list.append(hp.name)
    hyperparameters.values = hps.values
    if extend_list != []:
        return hyperparameters,extend_list # this operation does extend the hyperspace
    return hyperparameters,None # whether this operation extend the hyperspace


def build_train(hyperparameters,hm,key,new_value_list,config,root_save_dir,score_dict_path,score_dict,tmp_save_dir,previous=None):
    score_dict[key]={}

    for value in new_value_list:
        new_hp=hyperparameters.copy()
        # new_hp.values[key]=value
        #new_hp,extend=get_new_hps(new_hp,key,value)#TODO: back
        extend=None# TODO: delete, only use in block_type test

        # if extend==None:
        # only test one operation
        new_save_dir=os.path.join(root_save_dir,'{}-{}'.format(key.split('/')[-1],str(value)))
        os.makedirs(new_save_dir)
        new_save_path=os.path.join(new_save_dir,'gradient_weight.pkl')
        new_hp_path=os.path.join(new_save_dir,'param.pkl')
        with open(new_hp_path, 'wb') as f:
            pickle.dump(new_hp, f)
        model=hm.build(new_hp)
        model=adapt(model)# need to preprocess for text model!!!
        config['callbacks']=[]
        config['callbacks'].append(
        LossHistory(training_data=config['dataset'],
            batch_size=config['batch_size'],
            model=model,
            total_epoch=config['epoch'],
            save_path=new_save_path,
            tmp_save_dir=tmp_save_dir
        )) 
        _,score=train_model(config=config,model=model,callbacks=config['callbacks'],save_dir=new_save_dir)
        # score=0#TODO:delete

        score_dict[key][str(value)]={}
        score_dict[key][str(value)]['hp_path']=new_hp_path
        score_dict[key][str(value)]['score']=score
        score_dict[key][str(value)]['diff']=score-score_dict['origin']['score']
        if previous!=None:
            score_dict[key][str(value)]['previous']=previous
        with open(score_dict_path, 'wb') as f:
            pickle.dump(score_dict, f)


def modify_model(model,acti=None,init=None,method='acti'):
    import sys
    sys.path.append('./utils/')
    import opt

    if method== 'acti':
        model_weight=model.get_weights()
        model=opt.modify_activations(model,acti)
        # model=opt.modify_initializer(model,b_initializer=init,k_initializer=init)
        try:
            model.set_weights(model_weight)
        except Exception as e:
            print(e)
    elif method=='init':
        model_weight=model.get_weights()
        model=opt.modify_initializer(model,b_initializer=init,k_initializer=init)#b_initializer=init,
        try:
            model.set_weights(model_weight)
        except Exception as e:
            print(e)
    return model

def get_combine_list(exist_path):
    with open(exist_path, 'rb') as f:#input,bug type,params
        exist_result = pickle.load(f)
    exist_result_keys=list(exist_result.keys())

    import itertools
    arch_list=['resnet', 'xception', 'vanilla', 'efficient']
    loss_list=['slow_converge','oscillating','normal']
    grad_list=['explode','vanish','normal']
    wgt_list=['nan_weight','normal']
    all_list = [arch_list, loss_list, grad_list, wgt_list]
    tmp_combine_list=list(itertools.product(*all_list))
    combine_list=[]
    for l in tmp_combine_list:
        k='{}-{}-{}-{}'.format(l[0],l[1],l[2],l[3])
        if k not in exist_result_keys:
            combine_list.append(k)
    return combine_list

def load_train(root_model_path,config,root_save_dir,score_dict_path,score_dict):
    acti_list=['selu','tanh']# relu
    init_list=['he_uniform','lecun_uniform']#,'glorot_uniform'

    score_dict['activation']={}
    for acti in acti_list:
        model=load_model(root_model_path,custom_objects=ak.CUSTOM_OBJECTS)

        opt=model.optimizer
        for key in model.loss.keys():
            loss=model.loss[key].name
            break
        
        new_model=modify_model(model,acti=acti,method='acti')

        new_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        new_save_dir=os.path.join(root_save_dir,'{}'.format(acti))
        new_hp={}
        new_hp['activation']=acti
        os.makedirs(new_save_dir)
        new_save_path=os.path.join(new_save_dir,'gradient_weight.pkl')
        new_hp_path=os.path.join(new_save_dir,'param.pkl')
        with open(new_hp_path, 'wb') as f:
            pickle.dump(new_hp, f)

        config['callbacks']=[]
        config['callbacks'].append(
        LossHistory(training_data=config['dataset'],
        batch_size=config['batch_size'],
        model=new_model,
        total_epoch=config['epoch'],
        save_path=new_save_path,
        )) 
        _,score=train_model(config=config,model=new_model,callbacks=config['callbacks'],save_dir=new_save_dir)

        score_dict['activation'][acti]={}
        score_dict['activation'][acti]['hp_path']=new_hp_path
        score_dict['activation'][acti]['score']=score
        score_dict['activation'][acti]['diff']=score-score_dict['origin']['score']
        with open(score_dict_path, 'wb') as f:
            pickle.dump(score_dict, f)
    
    score_dict['initial']={}
    for init in init_list:
        model=load_model(root_model_path,custom_objects=ak.CUSTOM_OBJECTS)
        
        opt=model.optimizer
        for key in model.loss.keys():
            loss=model.loss[key].name
            break
        
        new_model=modify_model(model,acti=acti,method='acti')

        new_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        new_save_dir=os.path.join(root_save_dir,'{}'.format(init))
        new_hp={}
        new_hp['initial']=init
        os.makedirs(new_save_dir)
        new_save_path=os.path.join(new_save_dir,'gradient_weight.pkl')
        new_hp_path=os.path.join(new_save_dir,'param.pkl')
        with open(new_hp_path, 'wb') as f:
            pickle.dump(new_hp, f)

        config['callbacks']=[]
        config['callbacks'].append(
        LossHistory(training_data=config['dataset'],
        batch_size=config['batch_size'],
        model=new_model,
        total_epoch=config['epoch'],
        save_path=new_save_path,
        )) 
        _,score=train_model(config=config,model=new_model,callbacks=config['callbacks'],save_dir=new_save_dir)

        score_dict['initial'][init]={}
        score_dict['initial'][init]['hp_path']=new_hp_path
        score_dict['initial'][init]['score']=score
        score_dict['initial'][init]['diff']=score-score_dict['origin']['score']
        with open(score_dict_path, 'wb') as f:
            pickle.dump(score_dict, f)

# tf.data.experimental.save(
#     ds, tf_data_path, compression='GZIP'
# )
# with open(tf_data_path + '/element_spec', 'wb') as out_:  # also save the element_spec to disk for future loading
#     pickle.dump(ds.element_spec, out_)

# for text dataset:
def load_zipped_imdb(dirs='/home/zxy/main/DL_autokeras/test_codes/FORM/imdb_dataset'):
    with open(dirs + '/element_spec', 'rb') as in_:
        es = pickle.load(in_)
    new_dataset = tf.data.experimental.load(
        dirs, es, compression='GZIP')
    return new_dataset

def adapt(model, dataset_name='imdb'):
    if dataset_name=='imdb':
        dataset=load_zipped_imdb()
    elif dataset_name=='reuters':
        dataset=load_zipped_imdb('/home/zxy/main/DL_autokeras/test_codes/FORM/reuters_dataset')
    else:
        print('not support dataset')
        os._exit(0)
    x = dataset.map(lambda x, y: x)

    def get_output_layer(tensor):
        tensor = nest.flatten(tensor)[0]
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            if not isinstance(layer, preprocessing.PreprocessingLayer):
                break
            input_node = nest.flatten(layer.input)[0]
            if input_node is tensor:
                return layer
        return None

    for index, input_node in enumerate(nest.flatten(model.input)):
        temp_x = x.map(lambda *args: nest.flatten(args)[index])
        layer = get_output_layer(input_node)
        while layer is not None:
            if isinstance(layer, preprocessing.PreprocessingLayer):
                layer.adapt(temp_x)
            temp_x = temp_x.map(layer)
            layer = get_output_layer(layer.output)
    return model