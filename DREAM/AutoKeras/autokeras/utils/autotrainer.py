# AutoTrainer: a tool to predict the performance of model;
# If the current structure have potential training problems, it will be stopped by autotrainer
# Future work will add the autofix part.


import os
import sys
import psutil
import csv
import numpy as np
import pickle
import keras
from keras.models import load_model,Sequential
import keras.backend as K
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import copy

from autokeras.utils import monitor as mn
from autokeras.utils import compute_gradient as cg
from autokeras.utils import repair as rp

temp_fix_saved_dir='/data/zxy/DL_autokeras/1Autokeras/test_codes/experiment/autokeras_log/'

default_param = {'beta_1': 1e-3,
                 'beta_2': 1e-4,
                 'beta_3': 70,
                 'gamma': 0.7,
                 'zeta': 0.03,
                 'eta': 0.2,
                 'delta': 0.01,
                 'alpha_1': 0,
                 'alpha_2': 0,
                 'alpha_3': 0,
                 'Theta': 0.55
                 }

def read_data(dataset,batch_size):
    # read data from a new unzipped dataset.
    trainX=dataset['x'][:batch_size]
    trainy=dataset['y'][:batch_size]
    return trainX,trainy

def check_training(path):
    if not os.path.exists(path):
        return False,None
    else:
        with open(path, 'rb') as f:  #input,bug type,params
            output = pickle.load(f)
        problem = output['issue_list']
        if problem!=[]:
            return True,problem
        return False,None

def update_history(trial,old_history,new_history):
    # old history(struct) contains epoch,model,history,and othe message.
    # new history(dict) only contains history.
    old_history.history=new_history
    new_epoch=list(range(0,len(new_history['val_accuracy'])))
    old_history.epoch=new_epoch
    # trial.metrics.metrics['loss']._observations[0].value[0]
    for key in new_history.keys():
        origin_element=copy.deepcopy(trial.metrics.metrics[key]._observations[0])
        for ep in new_epoch:
            trial.metrics.metrics[key]._observations[ep]=copy.deepcopy(origin_element)
            trial.metrics.metrics[key]._observations[ep].step=ep
            trial.metrics.metrics[key]._observations[ep].value[0]=new_history[key][ep]

    return old_history,trial

def hyper_bn(hyperparameters):
    for key in hyperparameters.values.keys():
        if 'use_batchnorm' in key:
            hyperparameters.values[key]=True
    return hyperparameters


def update_hyper_params(hyperparameters,new_config,modify):
    if modify=='bn':
        hyperparameters=hyper_bn(hyperparameters)

    new_opt_lr=K.eval(new_config['opt'].lr)
    new_opt_name=new_config['opt'].get_config()['name'].lower()
    if new_opt_name in ["adam", "sgd", "adam_weight_decay"]:
        hyperparameters.values['optimizer']=new_opt_name
    hyperparameters.values['learning_rate']=float(new_opt_lr)
    #1.25 加入针对其他异常的判断
    return hyperparameters

class LossHistory(keras.callbacks.Callback):

    def __init__(self,training_data,batch_size,model,total_epoch,determine_threshold=2,satisfied_acc=0.7,\
        checktype='epoch_1',params={}): #only support epoch method now
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
        self.trainX,self.trainy = read_data(training_data,batch_size)
        self.batch_size=batch_size
        self.model=model
        self.satisfied_acc=satisfied_acc
        self.count=0
        self.checktype=checktype.split('_')[0]
        self.checkgap=int(checktype.split('_')[-1])
        self.issue_list=[]
        self.total_epoch=total_epoch
        self.determine_threshold=determine_threshold
        self.params=params
        if self.params=={}:
            self.params=default_param

        self.history={}
        self.history['loss']=[]
        self.history['acc']=[]
        self.history['val_loss']=[]
        self.history['val_acc']=[]

        self.Monitor=mn.IssueMonitor(total_epoch,self.satisfied_acc,self.params,self.determine_threshold)

    def on_train_begin(self,logs=None):
        weights=self.model.trainable_weights# get trainable weights
        if not cg.check_version(tf.__version__):
            try:
                grads = self.model.optimizer.get_gradients(self.model.total_loss, weights)
                symb_inputs = [self.model._feed_inputs , self.model._feed_targets , self.model._feed_sample_weights,K.learning_phase()]#input,corresponding label,weight of each sample(all of them are 1),learning rate(we set it to 0)
                self.f = K.function(symb_inputs, grads)
                self.new_computation=False
            except:
                self.new_computation=True
        else:
            self.new_computation=True       


    def on_epoch_end(self,epoch,logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_acc'].append(logs.get('val_accuracy'))
        if (epoch)%self.checkgap==0:
            
            trainingExample = self.trainX
            trainingY=self.trainy
            if self.new_computation==False:
                x, y, sample_weight = self.model._standardize_user_data(trainingExample, trainingY)
                #output_grad = f(x + y + sample_weight)
                self.evaluated_gradients = self.f([x , y , sample_weight,0])
            else:
                try:
                    self.evaluated_gradients = cg.get_gradients(self.model, trainingExample, trainingY)
                except:
                    layer_name = self.model.layers[3].name
                    # find embedding layer
                    self.evaluated_gradients = cg.rnn_get_gradients(self.model,layer_name, trainingExample, trainingY)
                # x, y, sample_weight = self.model._standardize_user_data(trainingExample, trainingY)
                # #output_grad = f(x + y + sample_weight)
                # self.evaluated_gradients = self.f([x , y , sample_weight,0])
            gradient_list=[]
            for i in range(len(self.evaluated_gradients)):
                if isinstance(self.evaluated_gradients[i],np.ndarray):
                    gradient_list.append(self.evaluated_gradients[i])

            self.issue_list=self.Monitor.determine(self.model,self.history,gradient_list,self.checkgap)

            self.evaluated_gradients=0
            gradient_list=0

            if self.issue_list!=[]:
                self.issue_list=list(set(self.issue_list))
                self.model.stop_training = True
                print('\nThis Model have potential training problem:',self.issue_list)
                print('Stop current training and search for next model')


    def on_train_end(self,logs=None):
        log_path='/data/zxy/DL_autokeras/1Autokeras/test_codes/experiment/autokeras_log/issue_history.pkl'
        tmpset={'issue_list':self.issue_list,'history':self.history}
        with open(log_path, 'wb') as f:
            pickle.dump(tmpset, f)
        print('Finished Training')

def fix_model(model,data,batch_size,epoch,problem_list,add_callbacks):
    train_config = unpack_train_config(model,data,epoch,batch_size,add_callbacks)
    rm = rp.Repair_Module(
        model=model,
        train_config=train_config,
        issue_list=problem_list,
        satisfied_acc=default_param['Theta'],
        determine_threshold=5,

    )  #train_config need to be packed and issue need to be read.
    result,model,history,issue_list,now_issue,new_config,modify = rm.solve()# all message will be save in a tmp dir
    return model,history,new_config,modify

def unpack_train_config(model,data,epoch,batch_size,add_callbacks=[]):
    config_dict={}
    config_dict['opt']=model.optimizer
    for key in model.loss.keys():
        config_dict['loss']=model.loss[key].name
        break
    config_dict['dataset']=data
    config_dict['batch_size']=batch_size
    config_dict['epoch']=epoch
    config_dict['callbacks']=add_callbacks
    return config_dict

def extract_dataset(data_x,data_val):
    dataset={}
    dataset['x']=[]
    dataset['y']=[]
    dataset['x_val']=[]
    dataset['y_val']=[]
    batch_size=data_x[0][0].shape[0]
    # if data_format==True:
    for i in data_x:
        try:
            _=i[0].shape[1]
            tmp_i=i[0]
        except:
            tmp_i=i[0].reshape((-1,1))
      
        if dataset['x']==[]:
            dataset['x']=tmp_i
        else:
            try:
                dataset['x']=np.row_stack((dataset['x'],tmp_i))
            except:
                pass
        if dataset['y']==[]:
            dataset['y']=i[1]
        else:
            try:
                dataset['y']=np.vstack((dataset['y'],i[1]))
            except:
                pass                
    for j in data_val:
        try:
            _=j[0].shape[1]
            tmp_j=j[0]
        except:
            tmp_j=j[0].reshape((-1,1))

        if dataset['x_val']==[]:
            dataset['x_val']=tmp_j
        else:
            try:
                dataset['x_val']=np.row_stack((dataset['x_val'],tmp_j))
            except:
                pass
        if dataset['y_val']==[]:
            dataset['y_val']=j[1]
        else:
            try:
                dataset['y_val']=np.vstack((dataset['y_val'],j[1]))
            except:
                pass
    dataset['y']=dataset['y'].reshape(-1,)
    dataset['y_val']=dataset['y_val'].reshape(-1,)
    return dataset,batch_size
    # return data,batch_size
    
    
def model_retrain(model,
                config,
                satisfied_acc,
                solution,
                determine_threshold,
                checktype):

    model.compile(loss=config['loss'], optimizer=config['opt'], metrics=['accuracy'])
    callback_bk=copy.deepcopy(config['callbacks'])
    config['callbacks'].append(
        LossHistory(training_data=config['dataset'],
        batch_size=config['batch_size'],
        model=model,
        total_epoch=config['epoch']))  

    callbacks_new=list(set(config['callbacks']))
    history = model.fit(config['dataset']['x'], config['dataset']['y'],batch_size=config['batch_size'], validation_data=(config['dataset']['x_val'], config['dataset']['y_val']),\
        epochs=config['epoch'],callbacks=callbacks_new)
    issue_path = os.path.join(temp_fix_saved_dir, 'issue_history.pkl')   
    with open(issue_path, 'rb') as f:  
        output = pickle.load(f)
    new_issues = output['issue_list']
    if 'need_train' in new_issues:
        new_issues=[]
    test_acc=history.history['val_accuracy'][-1]
    config['callbacks']=callback_bk
    return model,new_issues,test_acc,history.history