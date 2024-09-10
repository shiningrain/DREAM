# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
NNI example trial code.

- Experiment type: Hyper-parameter Optimization
- Trial framework: Tensorflow v2.x (Keras API)
- Model: LeNet-5
- Dataset: MNIST
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import logging

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import nni
from utils_data import get_reuters
from utils import *
import autokeras as ak
import kerastuner
import pickle
import uuid
import json
import time
import argparse
import sys
sys.path.append('../utils')
from operation_test_utils import adapt
import tensorflow_datasets as tfds
_logger = logging.getLogger('mnist_example')
_logger.setLevel(logging.INFO)



class ReportIntermediates(Callback):
    """
    Callback class for reporting intermediate accuracy metrics.

    This callback sends accuracy to NNI framework every 100 steps,
    so you can view the learning curve on web UI.

    If an assessor is configured in experiment's YAML file,
    it will use these metrics for early stopping.
    """
    def on_epoch_end(self, epoch, logs=None):
        """Reports intermediate accuracy to NNI framework"""
        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        # score_list=[10000-i for i in logs['val_loss']]
        # nni.report_intermediate_result(score_list)
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])

def get_new_hps(hyperparameters,param_dict):
    from kerastuner.engine import hyperparameters as hp_module# /kerastuner/engine/oracle.py line 395
    # while 1:
    hps = hp_module.HyperParameters()
    # Generate a set of random values.
    extend_list=[]
    for hp in hyperparameters.space:
        hps.merge([hp])
        if hps.is_active(hp):  # Only active params in `values`.
            # test for augmentations
            if hp.name in param_dict.keys():
                hps.values[hp.name]=param_dict[hp.name]
            else:
                hps.values[hp.name] = hp.random_sample(time.time())
    hyperparameters.values = hps.values
    return hyperparameters

def get_real_search_space(tmp_search_space):
    new_param_dict={}
    for key,value in tmp_search_space.items():
        if isinstance(value,dict):
            new_param_dict[key]=value['_name']
            for _vkey in value.keys():
                if _vkey=='_name':continue
                new_param_dict[_vkey]=value[_vkey]
        else:
            new_param_dict[key]=value
          
    # check trainable
    delete_flag=False
    for key1 in new_param_dict.keys():
        if 'trainable' in key1:
            for key2 in new_param_dict.keys():
                if 'pretrain' in key2:
                    if not new_param_dict[key2]:
                        delete_flag=True
                        break
        if delete_flag:
            del new_param_dict[key1]
            break
    return new_param_dict

def main(params_dict,
        #  hyperparam,
         epoch=10,
         batch_size=32,
         hm_path='./hypermodel-reuters.pkl',
         tmp_dir='./tmp',
         save_dir='./demo_result'):
    """
    Main program:
      - Build network
      - Prepare dataset
      - Train the model
      - Report accuracy to tuner
    """
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    log_path=os.path.join(save_dir,'log.pkl')
    gradient_save_path=os.path.join(tmp_dir,'gradient_weight.pkl')
    with open(log_path, 'rb') as f:
        log_dict = pickle.load(f)
           
    (x_train, y_train), (x_test, y_test)=get_reuters()

    y_train = np.array([example for example in y_train])
    y_train = np.eye(46)[y_train]
    y_test = np.array([example for example in y_test])
    y_test = np.eye(46)[y_test]
    _logger.info('Dataset loaded')
    
    with open(hm_path, 'rb') as f:
        hm = pickle.load(f)
    new_hp_value=get_real_search_space(params_dict)
    
    with open(log_dict['origin_param_path'], 'rb') as f:
        hyperparameters = pickle.load(f)

    new_hp=get_new_hps(hyperparameters,new_hp_value)

    print('============'+str(log_dict['cur_trial'])+'=============')
    model = hm.build(new_hp)


    new_hp=get_new_hps(hyperparameters,new_hp_value)
    model = hm.build(new_hp)
    model=adapt(model,dataset_name='reuters')
    c=new_hp.values
    print(c==new_hp_value)
    _logger.info('Model built')

    callbacks=[]
    # callbacks.append(ReportIntermediates())
    callbacks.append(tf_callbacks.EarlyStopping(patience=10, min_delta=0))
    data,_=extract_dataset_np([x_train[:batch_size,...],y_train[:batch_size]],tmp_dir=tmp_dir,method='reuters')


    history=model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epoch,
        verbose=1,
        callbacks=callbacks,
        validation_data=(x_test, y_test)
    )
    _logger.info('Training completed')
    
    # loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    train_history=history.history
    max_val_acc=max(train_history['val_accuracy'])
    nni.report_final_result(max_val_acc)  # send final accuracy to NNI tuner and web UI
    print(max_val_acc)
    _logger.info('Final accuracy reported: %s', max_val_acc)
    

    model_path=os.path.join(save_dir,'best_model.h5py')
    best_param_path=os.path.join(save_dir,'best_param.pkl')
    
    if not os.path.exists(log_path):
        log_dict={}
        log_dict['cur_trial']=-1
    else:
        log_dict['cur_trial']+=1
        current_time=time.time()-log_dict['start_time']
    
    log_dict[log_dict['cur_trial']]={}
    log_dict[log_dict['cur_trial']]['time']=current_time
    log_dict[log_dict['cur_trial']]['history']=train_history
    log_dict[log_dict['cur_trial']]['score']=max_val_acc
    log_dict[log_dict['cur_trial']]['hp_value']=new_hp.values
    if 'best_score' in log_dict.keys():
        if max_val_acc>log_dict['best_score']:
            log_dict['best_score']=max_val_acc
            model.save(model_path,save_format='tf')
            with open(best_param_path, 'wb') as f:
                pickle.dump(new_hp, f)
    else:
        log_dict['best_score']=max_val_acc
        model.save(model_path,save_format='tf')
        with open(best_param_path, 'wb') as f:
            pickle.dump(new_hp, f)
    with open(log_path, 'wb') as f:
        pickle.dump(log_dict, f)

def convert_search_space(hyperparams,search_space_path='./search_space-text-new.json'):
    def covert_dict(_dict):
        _output_dict={}
        for key,value in _dict.items():
            if key=='_name' or key not in hyperparams_name: continue
            if isinstance(value['_value'][0],dict):
                tmp_index_list=[i['_name'] for i in value['_value']]
                _index=tmp_index_list.index(hyperparams.values[key])
                _output_dict[key]={'_name':value['_value'][_index]['_name']}
                _tmp_output_dict=covert_dict(value['_value'][_index])
                for sup_key,sup_value in _tmp_output_dict.items():
                    if sup_key not in _output_dict[key].keys():
                        _output_dict[key][sup_key]=sup_value
            else:
                _output_dict[key]=value['_value'][value['_value'].index(hyperparams.values[key])]
        return _output_dict
    
    hyperparams_name=list(hyperparams.values.keys())
    with open(search_space_path, 'r') as f:
        search_space = json.load(f)
    unformatted_parameters=covert_dict(search_space)
    return unformatted_parameters

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test image classification')
    parser.add_argument('--save_dir','-sd',default='./demo_result-reuters-0', help='searching result path')
    # parser.add_argument('--source_param','-prm',default='./param.pkl', type=str, help='source_param path')
    parser.add_argument('--tmp_dir','-td',default='./tmp-text', type=str, help='searching result path')
    parser.add_argument('--epoch','-ep',default=10, type=int, help='epoch')
    parser.add_argument('--batch_size','-bs',default=32, type=int, help='batch_size')
    parser.add_argument('--hypermodel_path','-hp',default='./hypermodel-reuters.pkl', type=str, help='hypermodel_path')

    args = parser.parse_args()
    
    
    params_dict={}
        

    tuned_params = nni.get_next_parameter()
    params_dict.update(tuned_params)
    print(params_dict)

    
    main(params_dict,
        #  hyperparameters,
         save_dir=args.save_dir,
         tmp_dir=args.tmp_dir,
         batch_size=args.batch_size,
         epoch=args.epoch,
         hm_path=args.hypermodel_path)
