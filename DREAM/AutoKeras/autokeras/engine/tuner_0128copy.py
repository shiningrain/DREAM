# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        save_dict={}
        save_dict['gradient']={}
        save_dict['weight']={}
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)

        self.x_path='./Test_dir/tmp/x.npy'
        self.y_path='./Test_dir/tmp/y.npy'
        self.model_path='./Test_dir/tmp/model.h5'
        trainingExample = self.trainX
        trainingY=self.trainy
        np.save('./Test_dir/tmp/x.npy',trainingExample)
        np.save('./Test_dir/tmp/y.npy',trainingY)


    def on_epoch_end(self,epoch,logs={}):
        try:
            self.model.save(self.model_path)
        except:
            import time
            time.sleep(5)
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
            self.model.save(self.model_path)
        get_gradient(self.model_path,self.x_path,self.y_path,epoch,self.save_path)

    # def on_train_end(self,logs=None):
    #     print('Finished Training')


def get_gradient(model_path,x_path,y_path,epoch,save_path):
    import subprocess
    command="/data/zxy/anaconda3/envs/autokeras_test/bin/python ./utils/get_gradient_on_cpu.py -m {} -dx {} -dy {} -ep {} -sp {}" #TODO:need to set your your python interpreter path

    out_path=save_path.split('.')[0]+'_out'
    out_file = open(out_path, 'w')
    out_file.write('logs\n')
    run_cmd=command.format(model_path,x_path,y_path,epoch,save_path)
    subprocess.Popen(run_cmd, shell=True, stdout=out_file, stderr=out_file)

class AutoTuner(kerastuner.engine.tuner.Tuner):
    """A Tuner class based on KerasTuner for AutoKeras.

    Different from KerasTuner's Tuner class. AutoTuner's not only tunes the
    Hypermodel which can be directly built into a Keras model, but also the
    preprocessors. Therefore, a HyperGraph stores the overall search space containing
    both the Preprocessors and Hypermodel. For every trial, the HyperGraph build the
    PreprocessGraph and KerasGraph with the provided HyperParameters.

    The AutoTuner uses EarlyStopping for acceleration during the search and fully
    train the model with full epochs and with both training and validation data.
    The fully trained model is the best model to be used by AutoModel.

    # Arguments
        oracle: kerastuner Oracle.
        hypermodel: kerastuner KerasHyperModel.
        **kwargs: The args supported by KerasTuner.
    """

    def __init__(self, oracle, hypermodel, **kwargs):
        # Initialize before super() for reload to work.
        self._finished = False
        super().__init__(oracle, hypermodel, **kwargs)
        # Save or load the HyperModel.
        self.hypermodel.hypermodel.save(os.path.join(self.project_dir, "graph"))
        self.hyper_pipeline = None

    def _populate_initial_space(self):
        # Override the function to prevent building the model during initialization.
        return

    def get_best_model(self):
        with hm_module.maybe_distribute(self.distribution_strategy):
            model = tf.keras.models.load_model(self.best_model_path)
        return model

    def get_best_pipeline(self):
        return pipeline_module.load_pipeline(self.best_pipeline_path)

    def _pipeline_path(self, trial_id):
        return os.path.join(self.get_trial_dir(trial_id), "pipeline")

    def _prepare_model_build(self, hp, **kwargs):
        """Prepare for building the Keras model.

        It build the Pipeline from HyperPipeline, transform the dataset to set
        the input shapes and output shapes of the HyperModel.
        """
        dataset = kwargs["x"]
        pipeline = self.hyper_pipeline.build(hp, dataset)
        pipeline.fit(dataset)
        dataset = pipeline.transform(dataset)
        self.hypermodel.hypermodel.set_io_shapes(data_utils.dataset_shape(dataset))

        if "validation_data" in kwargs:
            validation_data = pipeline.transform(kwargs["validation_data"])
        else:
            validation_data = None
        return pipeline, dataset, validation_data

    def evaluate_layer_trainable(self,model):
        for l in model.layers:
            try:
                sub_layer_list=l.layers
                for sl in sub_layer_list:
                    print('SubLayer "{}" :{}'.format(sl.name,sl.trainable))
            except Exception as e:
                print('Layer "{}" :{}'.format(l.name,l.trainable))
    
    def unfreeze_model(self,model,method='part'):
        part_list=['block1','block2','block3','block4']
        for l in model.layers:
            try:
                sub_layer_list=l.layers
                for sl in sub_layer_list:
                    if method=='part':
                        for blk in part_list:
                            if (blk in sl.name) or (isinstance(sl, tf.keras.layers.BatchNormalization)):
                                sl.trainable = False
                                break
                            else:
                                sl.trainable = True
                                
                    else:
                        # if not isinstance(sl, tf.keras.layers.BatchNormalization):
                        sl.trainable = True
            except Exception as e:
                if not isinstance(l, tf.keras.layers.BatchNormalization):
                    l.trainable = True
        return model
    
    def freeze_model(self,model,method='step_1_freeze'):
        # freeze model method: all, no. not provide 'bn' ,'part' now.
        if method=='no':
            model.trainable=True
            
        elif method=='all':
            for l in range(len(model.layers)):
                try:
                    sub_layer_list=model.layers[l].layers
                    model.layers[l].trainable=False # only freeze the main part of model
                except Exception as e:
                    pass
        elif method=='bn':
            for l in range(len(model.layers)):
                try:
                    sub_layer_list=model.layers[l].layers
                    for sl in range(len(sub_layer_list)):
                        if (isinstance(model.layers[l].layers[sl], tf.keras.layers.BatchNormalization)):
                            model.layers[l].layers[sl].trainable = False
                except Exception as e:
                    pass
            
        return model

    def freeze_fit(self, model, batch_size, freeze_status, **fit_kwargs):#,save_dir,opt,method='normal',
        # freeze and train
        if freeze_status==False :#or fit_kwargs['epochs']<10:# TODO:back          
            model, history = utils.fit_with_adaptive_batch_size(
                model, batch_size, **fit_kwargs
            )
            # with open(os.path.join(save_dir,'normal_history.pkl'), 'wb') as f:
            #     pickle.dump(history.history, f)
        else:
            # step1: freeze pretrain model
            hp=self.oracle.hyperparameters
            step_1_ratio = hp.Choice(
            "step_1_ratio", [0.1,0.2,0.3,0.4,0.5], default=0.2,
            parent_name='multi_step', parent_values=True
            )
            step_2_lr_scale = hp.Choice(
            "step_2_lr_scale", [1.0,0.1,0.01], default=0.1,
            parent_name='multi_step', parent_values=True
            )
            step_1_freeze = hp.Choice(
            "step_1_freeze", ['all','no','bn'], default='all',#,'part',
            parent_name='multi_step', parent_values=True
            )
            
            epoch1=int(fit_kwargs['epochs']*step_1_ratio)#int(fit_kwargs['epochs']/3)
            epoch2=fit_kwargs['epochs']-epoch1
            
            # with open(os.path.join('/data/zxy/DL_autokeras/1Autokeras/test_codes/experiment/FORM/Test_dir','tmp_hp.pkl'), 'wb') as f:
            #     pickle.dump(hp, f)
            fit_kwargs['epochs']=epoch1
            origin_lr=copy.deepcopy(hp.values['learning_rate'])
            origin_optimizer=copy.deepcopy(hp.values['optimizer'])
            hp.values['learning_rate']=hp.values['learning_rate']/step_2_lr_scale
            hp.values['optimizer']='adam'
            
            # TODO: add freeze part model.
            model=self.freeze_model(model,method=step_1_freeze)
            
            # self.evaluate_layer_trainable(model)
            
            model=self.hypermodel.build_optimizer(hp,model)
            
            model, history = utils.fit_with_adaptive_batch_size(
                model, batch_size, **fit_kwargs
            )
            
            
            # with open(os.path.join(save_dir,'unfreeze_history_1.pkl'), 'wb') as f:
            #     pickle.dump(history.history, f)
            
            # step2: unfreeze whole model and freeze bn layer
            
            # epoch3=fit_kwargs['epochs'] 

            fit_kwargs['epochs']=epoch2
            hp.values['learning_rate']=origin_lr
            hp.values['optimizer']=origin_optimizer
            
            model.trainable=True
            model=self.hypermodel.build_optimizer(hp,model)            
            
            model, history_new = utils.fit_with_adaptive_batch_size(
                model, batch_size, **fit_kwargs
            )

            for k in history.history.keys():     
                history.history[k]+=history_new.history[k]
            # with open(os.path.join(save_dir,'unfreeze_history_2.pkl'), 'wb') as f:
            #     pickle.dump(history.history, f)
                
            # if method=='bn':
            #     model.trainable=True
            # fit_kwargs['epochs']=epoch3
            # if opt=='nadam':
            #     optimizer = tf.keras.optimizers.Nadam(learning_rate=0.00001, beta_1=0.9)#model.optimizer#
            # elif opt=='adam':
            #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)#model.optimizer#
            # else:
            #     optimizer = model.optimizer
            # # optimizer = tf.keras.optimizers.Nadam(learning_rate=0.00001, beta_1=0.9)#model.optimizer#
            # model.compile(
            #     optimizer=optimizer, loss=model.loss, metrics=["accuracy"]
            # )
            
            # model, history = utils.fit_with_adaptive_batch_size(
            #     model, batch_size, **fit_kwargs
            # )
            # with open(os.path.join(save_dir,'unfreeze_history_3.pkl'), 'wb') as f:
            #     pickle.dump(history.history, f)
            
        return model,history


    # 0106 old freeze_fit
    # def freeze_fit(self, model, batch_size, freeze_status,save_dir,opt,method='normal', **fit_kwargs):
    #     # freeze and train
    #     if freeze_status==False or fit_kwargs['epochs']<10:
        
    #         # import tensorflow_addons as tfa
    #         # optimizer = tfa.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.01)
    #         # model.compile(
    #         #     optimizer=optimizer, loss=model.loss, metrics=["accuracy"]
    #         # )
    #         # print(1)
            
    #         model, history = utils.fit_with_adaptive_batch_size(
    #             model, batch_size, **fit_kwargs
    #         )
    #         with open(os.path.join(save_dir,'normal_history.pkl'), 'wb') as f:
    #             pickle.dump(history.history, f)
    #     else:
    #         # step1: freeze pretrain model
    #         epoch1=int(fit_kwargs['epochs']/6)#int(fit_kwargs['epochs']/3)
    #         bk_epoch=copy.deepcopy(fit_kwargs['epochs'])
    #         bk_opti=copy.deepcopy(model.optimizer)
    #         fit_kwargs['epochs']=epoch1
            
    #         if opt=='nadam':
    #             optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9)#model.optimizer#
    #         elif opt=='adam':
    #             optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)#model.optimizer#
    #         else:
    #             optimizer = model.optimizer
    #         # optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9)#model.optimizer#
            
            
            
    #         self.evaluate_layer_trainable(model)
            
    #         # # TODO:delete
    #         # model.trainable=True
            
    #         # model.compile(
    #         #     optimizer=bk_opti, loss=model.loss, metrics=["accuracy"]
    #         # )

    #         # /data/zxy/anaconda3/envs/autokeras_test/lib/python3.7/site-packages/autokeras/graph.py
    #         model.compile(
    #             optimizer=optimizer, loss=model.loss, metrics=["accuracy"]
    #         )
    #         model, history = utils.fit_with_adaptive_batch_size(
    #             model, batch_size, **fit_kwargs
    #         )
            
            
    #         with open(os.path.join(save_dir,'unfreeze_history_1.pkl'), 'wb') as f:
    #             pickle.dump(history.history, f)
            
    #         # step2: unfreeze whole model and freeze bn layer
            
    #         # epoch3=fit_kwargs['epochs'] 
    #         epoch2=bk_epoch-epoch1

    #         fit_kwargs['epochs']=epoch2
            
    #         # model=self.unfreeze_model(model,method='all')
    #         model.trainable=True
            
    #         # for l in model.layers:
    #         #     l.trainable = True
    #         #     try:
    #         #         start_train=False
    #         #         for sl in l.layers:
    #         #             if method=='bn' and isinstance(sl, tf.keras.layers.BatchNormalization):
    #         #                 sl.trainable = False
    #         #                 # pass # not block bn now, just unfreeze part of model
    #         #             else:
    #         #                 # for p in part_list:
    #         #                 if not start_train and 'block5' in sl.name:
    #         #                     # sl.trainable = True
    #         #                     start_train=True
    #         #                 if not start_train:
    #         #                     sl.trainable = False
    #         #     except Exception as e:
    #         #         pass

    #         self.evaluate_layer_trainable(model)
    #         if opt=='nadam':
    #             optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001, beta_1=0.9)#model.optimizer#
    #         elif opt=='adam':
    #             optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)#model.optimizer#\
    #         else:
    #             optimizer = model.optimizer
    #         # optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001, beta_1=0.9)#model.optimizer#
    #         model.compile(
    #             optimizer=bk_opti, loss=model.loss, metrics=["accuracy"]# TODO: back
    #         )

    #         model, history = utils.fit_with_adaptive_batch_size(
    #             model, batch_size, **fit_kwargs
    #         )
    #         with open(os.path.join(save_dir,'unfreeze_history_2.pkl'), 'wb') as f:
    #             pickle.dump(history.history, f)
                
    #         # if method=='bn':
    #         #     model.trainable=True
    #         # fit_kwargs['epochs']=epoch3
    #         # if opt=='nadam':
    #         #     optimizer = tf.keras.optimizers.Nadam(learning_rate=0.00001, beta_1=0.9)#model.optimizer#
    #         # elif opt=='adam':
    #         #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)#model.optimizer#
    #         # else:
    #         #     optimizer = model.optimizer
    #         # # optimizer = tf.keras.optimizers.Nadam(learning_rate=0.00001, beta_1=0.9)#model.optimizer#
    #         # model.compile(
    #         #     optimizer=optimizer, loss=model.loss, metrics=["accuracy"]
    #         # )
            
    #         # model, history = utils.fit_with_adaptive_batch_size(
    #         #     model, batch_size, **fit_kwargs
    #         # )
    #         # with open(os.path.join(save_dir,'unfreeze_history_3.pkl'), 'wb') as f:
    #         #     pickle.dump(history.history, f)
            
    #     return model,history

    # 3 step training freeze-unfreeze-low lr
    # def freeze_fit(self, model, batch_size, freeze_status,save_dir, **fit_kwargs):
    #     # freeze and train
    #     if freeze_status==False or fit_kwargs['epochs']<10:
    #         model, history = utils.fit_with_adaptive_batch_size(
    #             model, batch_size, **fit_kwargs
    #         )
    #         with open(os.path.join(save_dir,'normal_history.pkl'), 'wb') as f:
    #             pickle.dump(history.history, f)
    #     else:
    #         # step1: freeze pretrain model
    #         epoch1=int(fit_kwargs['epochs']/3)
    #         bk_epoch=copy.deepcopy(fit_kwargs['epochs'])
    #         fit_kwargs['epochs']=epoch1
    #         optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9)#model.optimizer#
    #         self.evaluate_layer_trainable(model)
            
    #         # /data/zxy/anaconda3/envs/autokeras_test/lib/python3.7/site-packages/autokeras/graph.py
    #         model.compile(
    #             optimizer=optimizer, loss=model.loss, metrics=["accuracy"]
    #         )
    #         model, history = utils.fit_with_adaptive_batch_size(
    #             model, batch_size, **fit_kwargs
    #         )

    #         with open(os.path.join(save_dir,'unfreeze_history_1.pkl'), 'wb') as f:
    #             pickle.dump(history.history, f)

    #         # step2: unfreeze hole model
    #         epoch2=fit_kwargs['epochs']
    #         epoch3=bk_epoch-epoch1-epoch2
    #         fit_kwargs['epochs']=epoch2

    #         # model=self.unfreeze_model(model,method='all')
    #         model.trainable=True
    #         self.evaluate_layer_trainable(model)
    #         optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001, beta_1=0.9)#model.optimizer#
    #         model.compile(
    #             optimizer=optimizer, loss=model.loss, metrics=["accuracy"]
    #         )

    #         model, history = utils.fit_with_adaptive_batch_size(
    #             model, batch_size, **fit_kwargs
    #         )
    #         with open(os.path.join(save_dir,'unfreeze_history_2.pkl'), 'wb') as f:
    #             pickle.dump(history.history, f)
                
    #         optimizer = tf.keras.optimizers.Nadam(learning_rate=0.00001, beta_1=0.9)#model.optimizer#
    #         model.compile(
    #             optimizer=optimizer, loss=model.loss, metrics=["accuracy"]
    #         )
            
    #         model, history = utils.fit_with_adaptive_batch_size(
    #             model, batch_size, **fit_kwargs
    #         )
    #         with open(os.path.join(save_dir,'unfreeze_history_3.pkl'), 'wb') as f:
    #             pickle.dump(history.history, f)
            
    #     return model,history

    def _build_and_fit_model_greedy(self, trial, fit_args, fit_kwargs):
        (
            pipeline,
            fit_kwargs["x"],
            fit_kwargs["validation_data"],
        ) = self._prepare_model_build(trial.hyperparameters, **fit_kwargs)
        pipeline.save(self._pipeline_path(trial.trial_id))


        import uuid
        import os
        import time
        import pickle
        import sys
        import shutil
        sys.path.append('./utils')
        from load_test_utils import traversalDir_FirstDir,read_opt,write_algw,evaluate_layer_trainable,modify_hp_value
        # root_path='./Test_dir/demo_result'
        root_path=fit_kwargs['root_path']
        del fit_kwargs['root_path']
        root_path=os.path.abspath(root_path)
        # get greedy ak path:
        origin_path=os.path.join(root_path,'origin')
        log_path=os.path.join(root_path,'log.pkl')
        with open(log_path, 'rb') as f:#input,bug type,params
            log_dict = pickle.load(f)
        params_path=log_dict['param_path']
             
        if traversalDir_FirstDir(root_path)==[]:
            with open(params_path, 'rb') as f:#input,bug type,params
                ak_params = pickle.load(f)
            
            with open('/data/zxy/DL_autokeras/1Autokeras/test_codes/experiment/autokeras_log/autokeras_1227_bayes/1227motihp.pkl', 'rb') as f:#input,bug type,params
                tmp_value = pickle.load(f)
            ak_params.values=tmp_value
            # zxy TODO: back
            # if log_dict['opt']==0:
            #     pass
            # else:
            # ak_params.values['optimizer']='nadam'#log_dict['opt_type']
            #     ak_params.values['learning_rate']=log_dict['opt']
            # ak_params.values['learning_rate']=log_dict['opt']
            # ak_params.values['momentum']=0.9
            # zxy TODO: back
            # ak_params=modify_hp_value(ak_params,'/data/zxy/DL_autokeras/1Autokeras/test_codes/experiment/FORM/Test_dir/tmp_hp_list.pkl','/data/zxy/DL_autokeras/1Autokeras/test_codes/experiment/FORM/Test_dir/tmp_hp_num.pkl')
            # ak_params.values['end_learning_rate']=1e-5
            trial.hyperparameters=ak_params
            self.oracle.hyperparameters=ak_params

        elif log_dict['cur_trial']>=25 and time_usage>=86400:#64800:
        # elif time_usage>=7200 and log_dict['cur_trial']>=20:
            # origin_num=int(os.path.basename(os.path.dirname(origin_path)).split('_')[-1])

            # new_dir='{}_{}'.format(root_path,num)
            new_dir='{}_{}'.format(root_path,'FORM')
            os.rename(root_path,new_dir)
            os._exit(0)

        
        model = self.hypermodel.build(trial.hyperparameters)
        
        
        self.adapt(model, fit_kwargs["x"])

        

        read_path=os.path.abspath('./Test_dir/tmp/tmp_action_value.pkl')
        if os.path.exists(read_path):
            special=True
            model=read_opt(model)# check for special actions like activation and initializer
        else:
            special=False
        

        msg={}
        msg['epochs']=fit_kwargs['epochs']
        data,batch_size=extract_dataset(list(fit_kwargs['x'].as_numpy_iterator()),fit_kwargs['validation_data'].as_numpy_iterator(),method='cifar')
        # msg['data']=data
        msg['batch']=batch_size

        save_path=os.path.abspath('./Test_dir/tmp/gradient_weight.pkl')
        predictor=LossHistory(training_data=data,model=model,total_epoch=fit_kwargs['epochs'],batch_size=batch_size,save_path=save_path)
        fit_kwargs['callbacks'].append(predictor)
        # fit_kwargs['callbacks'].append(tf_callbacks.EarlyStopping(patience=10, min_delta=0))

        # only provide multistep for pretrain model.
        hpk_list=list(trial.hyperparameters.values.keys())
        multi_step=False
        for hpkey in hpk_list:
            if 'trainable'in hpkey:
                multi_step=trial.hyperparameters.Boolean("multi_step", default=False)
                # self.oracle.hyperparameters also update here
        # # TODO: back
        # fit_kwargs['callbacks']=[]
        self.oracle.hyperparameters=trial.hyperparameters
        _, history = self.freeze_fit(#log_dict['freeze']True,'bn'
            model, self.hypermodel.hypermodel.batch_size,multi_step, **fit_kwargs
        )
        # save_dir=root_path,method=log_dict['method'],opt=log_dict['opt_type'],
        
        # #train
        # fit_kwargs['callbacks']=[]
        # optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001, beta_1=0.9)#model.optimizer#
        # model.compile(
        #     optimizer=optimizer, loss=model.loss, metrics=["accuracy"]
        # )
        # _, history = self.freeze_fit(
        #     model, self.hypermodel.hypermodel.batch_size,False,save_dir=root_path, **fit_kwargs
        # )

        # # notrain
        # fit_kwargs['callbacks']=[]
        # _, history = self.freeze_fit(
        #     model, self.hypermodel.hypermodel.batch_size,True,save_dir=root_path, **fit_kwargs
        # )
        # os._exit(0)
        
        train_history=history.history
        max_val_acc=max(train_history['val_accuracy'])
        trial_num_path=os.path.join(os.path.dirname(root_path),'num.pkl')
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        if not os.path.exists(log_path):
            log_dict={}
            log_dict['cur_trial']=0
            
        else:
            with open(log_path, 'rb') as f:#input,bug type,params
                log_dict = pickle.load(f)
                log_dict['cur_trial']+=1
                current_time=time.time()-log_dict['start_time']

        new_dir_name=str(log_dict['cur_trial'])+'-'+str(round(max_val_acc,2))+str(uuid.uuid3(uuid.NAMESPACE_DNS,str(time.time())))[-13:]

        new_dir_path=os.path.join(root_path,new_dir_name)
        
        log_dict[new_dir_name]={}
        log_dict[new_dir_name]['time']=current_time
        log_dict[new_dir_name]['history']=train_history
        log_dict[new_dir_name]['score']=max_val_acc
        log_dict[new_dir_name]['values']=trial.hyperparameters.values

        with open(log_path, 'wb') as f:
            pickle.dump(log_dict, f)

        trial_id_path=os.path.join(root_path,'trial_id.pkl')
        if os.path.exists(trial_id_path):
            with open(trial_id_path, 'rb') as f:#input,bug type,params
                trial_id_dict = pickle.load(f)
        else:
            trial_id_dict={}
        trial_id_dict[trial.trial_id]=new_dir_path
        with open(trial_id_path, 'wb') as f:
            pickle.dump(trial_id_dict, f)   


        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)
        else:
            new_dir_name=str(log_dict['cur_trial'])+'-'+str(round(max_val_acc,2))+str(uuid.uuid3(uuid.NAMESPACE_DNS,str(time.time())))[-13:]
            new_dir_path=os.path.join(root_path,new_dir_name)
            os.makedirs(new_dir_path)
        
        model_path=os.path.join(new_dir_path,'model.h5')
        history_path=os.path.join(new_dir_path,'history.pkl')
        hyperparam_path=os.path.join(new_dir_path,'param.pkl')
        message_path=os.path.join(new_dir_path,'msg.pkl')
        hm_path = os.path.join(new_dir_path,'hypermodel.pkl')
        new_save_path=os.path.join(new_dir_path,'gradient_weight.pkl')

        if special:
            new_read_path=os.path.join(new_dir_path,'tmp_action_value.pkl')
            
            shutil.move(read_path,new_read_path)

        shutil.move(save_path,new_save_path)

        with open(hm_path, 'wb') as f:
            pickle.dump(self.hypermodel, f)   
        
        with open(message_path, 'wb') as f:
            pickle.dump(msg, f)

        try:
            model.save(model_path)
        except Exception as e:
            print(e)
            model.save(new_dir_path, save_format='tf')

        with open(history_path, 'wb') as f:
            pickle.dump(train_history, f)
        with open(hyperparam_path, 'wb') as f:
            pickle.dump(trial.hyperparameters, f)
        
        # print(1)
        # if current_time>=7200 and log_dict['cur_trial']>=20:

        write_algw(new_dir_path)
        #TODO: delete
        
        # os._exit(0)

        return history

    def _build_and_fit_model_beam(self, trial, fit_args, fit_kwargs):
        (
            pipeline,
            fit_kwargs["x"],
            fit_kwargs["validation_data"],
        ) = self._prepare_model_build(trial.hyperparameters, **fit_kwargs)
        pipeline.save(self._pipeline_path(trial.trial_id))


        import uuid
        import os
        import time
        import pickle
        import sys
        import shutil
        sys.path.append('./utils')
        from load_test_utils import traversalDir_FirstDir,read_opt,write_algw,verify_hp
        root_path='./Test_dir/demo_result'
        root_path=os.path.abspath(root_path)
        # get greedy ak path:
        gpus=tf.config.experimental.list_physical_devices('GPU')
        origin_path=os.path.join(root_path,'origin')
        log_path=os.path.join(root_path,'log.pkl')
        with open(log_path, 'rb') as f:#input,bug type,params
            log_dict = pickle.load(f)
        params_path=log_dict['param_path']

        msg={}
        msg['epochs']=fit_kwargs['epochs']
        data,batch_size=extract_dataset(list(fit_kwargs['x'].as_numpy_iterator()),fit_kwargs['validation_data'].as_numpy_iterator(),method='cifar')
        # msg['data']=data
        msg['batch']=batch_size
        save_path=os.path.abspath('./Test_dir/tmp/gradient_weight.pkl')

        log_dict['cur_trial']+=1
        
        if traversalDir_FirstDir(root_path)==[]:
            with open(params_path, 'rb') as f:#input,bug type,params
                ak_params = pickle.load(f)
            
            trial.hyperparameters=ak_params

            self.oracle.hyperparameters=ak_params

        candidate_path=os.path.join(root_path,'candidate.pkl')
        candidate_sign=False
        if os.path.exists(candidate_path):
            with open(candidate_path, 'rb') as f:#input,bug type,params
                candidate = pickle.load(f)
            candidate_sign=True
            for i in range(len(candidate)):
                if not isinstance(candidate[i][0][0],dict):
                    candidate_sign=False
                    break
                    
        if candidate_sign:
            beam_size=3
            if beam_size<=len(candidate):
                # new_x,new_validation_data=fit_kwargs['sub_dataset'],fit_kwargs['sub_validation_data']
                self.new_fit_kwargs=self.get_new_kwargs(fit_kwargs)
                # candidate=self.subset_train(trial,candidate,self.new_fit_kwargs,beam_size=beam_size)
                candidate_score_list=[]

                subtrain_result_path=os.path.join(root_path,'subtrain_result.pkl')
                if os.path.exists(subtrain_result_path):
                    with open(subtrain_result_path, 'rb') as f:#input,bug type,params
                        subtrain_result = pickle.load(f)
                else:
                    subtrain_result={}
                    subtrain_result['num']=0

                for i in range(len(candidate)):
                    hp_bk=copy.deepcopy(candidate[i][0][0])
                    trial.hyperparameters.values=copy.deepcopy(candidate[i][0][0])
                    self.oracle.hyperparameters.values=copy.deepcopy(candidate[i][0][0])
                    try:
                        model = self.hypermodel.build(trial.hyperparameters)
                    except Exception as e:
                        print(e)
                        error_path=os.path.join(root_path,'error.pkl')
                    
                        with open(error_path, 'wb') as f:
                            pickle.dump(trial.hyperparameters, f)
                        continue
                    self.oracle.update_space(trial.hyperparameters)

                    if (not verify_hp(trial.hyperparameters.values,hp_bk)) or (not verify_hp(self.oracle.hyperparameters.values,hp_bk)):
                        trial.hyperparameters.values=hp_bk
                        self.oracle.hyperparameters.values=hp_bk

                    self.adapt(model, self.new_fit_kwargs["x"])

                    #TODO: modify later
                    read_path=os.path.abspath('./Test_dir/tmp/tmp_action_value.pkl')
                    if os.path.exists(read_path):
                        special=True
                        model=read_opt(model)# check for special actions like activation and initializer
                    else:
                        special=False

                    _, history = utils.fit_with_adaptive_batch_size(
                        model, self.hypermodel.hypermodel.batch_size, **self.new_fit_kwargs
                    )

                    max_val_acc=max(history.history['val_accuracy'])
                    # max_val_acc=np.random.random()#TODO: back

                    candidate_score_list.append(max_val_acc)

                    subtrain_result[subtrain_result['num']]=(trial.hyperparameters.values,max_val_acc)
                    subtrain_result['num']+=1
                    with open(subtrain_result_path, 'wb') as f:
                        pickle.dump(subtrain_result, f)


                import heapq
                max_index_list = list(map(candidate_score_list.index, heapq.nlargest(beam_size, candidate_score_list)))
                new_candidate_list=[]
                for index in max_index_list:
                    new_candidate_list.append(candidate[index])
                candidate=copy.deepcopy(new_candidate_list)

            del fit_kwargs['sub_dataset']
            del fit_kwargs['sub_validation_data']
            for i in range(beam_size):
                # if 'block_type' in candidate[i][0][1]:
                #     print(2)
                hp_bk=copy.deepcopy(candidate[i][0][0])
                trial.hyperparameters.values=copy.deepcopy(candidate[i][0][0])
                self.oracle.hyperparameters.values=copy.deepcopy(candidate[i][0][0])
                try:
                    model = self.hypermodel.build(trial.hyperparameters)
                    self.oracle.update_space(trial.hyperparameters)
                    
                    if (not verify_hp(trial.hyperparameters.values,hp_bk)) or (not verify_hp(self.oracle.hyperparameters.values,hp_bk)):
                        trial.hyperparameters.values=hp_bk
                        self.oracle.hyperparameters.values=hp_bk

                except:
                    error_path=os.path.join(root_path,'error-1.pkl')
                    
                    with open(error_path, 'wb') as f:
                        pickle.dump(trial.hyperparameters, f)
                    os._exit(0)
                    # If fail to build model, save and exit.
                self.adapt(model, fit_kwargs["x"])

                #TODO: modify later
                read_path=os.path.abspath('./Test_dir/tmp/tmp_action_value.pkl')
                if os.path.exists(read_path):
                    special=True
                    model=read_opt(model)# check for special actions like activation and initializer
                else:
                    special=False


                predictor=LossHistory(training_data=data,model=model,total_epoch=fit_kwargs['epochs'],batch_size=batch_size,save_path=save_path)
                fit_kwargs['callbacks'].append(predictor)

                _, history = utils.fit_with_adaptive_batch_size(
                    model, self.hypermodel.hypermodel.batch_size, **fit_kwargs
                )

                train_history=history.history
                max_val_acc=max(train_history['val_accuracy'])
                trial_num_path=os.path.join(os.path.dirname(root_path),'num.pkl')

                current_time=time.time()-log_dict['start_time']
                new_dir_name=str(log_dict['cur_trial'])+'-'+str(round(max_val_acc,2))+str(uuid.uuid3(uuid.NAMESPACE_DNS,str(time.time())))[-13:]

                new_dir_path=os.path.join(root_path,new_dir_name)
                
                log_dict[new_dir_name]={}
                log_dict[new_dir_name]['time']=current_time
                log_dict[new_dir_name]['history']=train_history
                log_dict[new_dir_name]['score']=max_val_acc
                log_dict[new_dir_name]['values']=trial.hyperparameters.values
                log_dict[new_dir_name]['candidate']=candidate[i]

                with open(log_path, 'wb') as f:
                    pickle.dump(log_dict, f)

                # TODO: not use in beam search
                trial_id_path=os.path.join(root_path,'trial_id.pkl')
                if os.path.exists(trial_id_path):
                    with open(trial_id_path, 'rb') as f:#input,bug type,params
                        trial_id_dict = pickle.load(f)
                else:
                    trial_id_dict={}
                if isinstance(trial_id_dict,dict):
                    trial_id_dict[trial.trial_id]=new_dir_path
                    with open(trial_id_path, 'wb') as f:
                        pickle.dump(trial_id_dict, f)   


                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)
                else:
                    new_dir_name=str(log_dict['cur_trial'])+'-'+str(round(max_val_acc,2))+str(uuid.uuid3(uuid.NAMESPACE_DNS,str(time.time())))[-13:]
                    new_dir_path=os.path.join(root_path,new_dir_name)
                    os.makedirs(new_dir_path)
                
                model_path=os.path.join(new_dir_path,'model.h5')
                history_path=os.path.join(new_dir_path,'history.pkl')
                hyperparam_path=os.path.join(new_dir_path,'param.pkl')
                message_path=os.path.join(new_dir_path,'msg.pkl')
                hm_path = os.path.join(new_dir_path,'hypermodel.pkl')
                new_save_path=os.path.join(new_dir_path,'gradient_weight.pkl')
                action_path=os.path.join(new_dir_path,'action.pkl')

                if special:
                    new_read_path=os.path.join(new_dir_path,'tmp_action_value.pkl')
                    
                    shutil.move(read_path,new_read_path)

                shutil.move(save_path,new_save_path)

                with open(hm_path, 'wb') as f:
                    pickle.dump(self.hypermodel, f)   
                
                with open(message_path, 'wb') as f:
                    pickle.dump(msg, f)
                
                with open(action_path, 'wb') as f:
                    pickle.dump(candidate[i][0][1], f)

                try:
                    model.save(model_path)
                except Exception as e:
                    print(e)
                    model.save(new_dir_path, save_format='tf')

                with open(history_path, 'wb') as f:
                    pickle.dump(train_history, f)
                with open(hyperparam_path, 'wb') as f:
                    trial.hyperparameters.values=candidate[i][0][0]
                    pickle.dump(trial.hyperparameters, f)
                
                # print(1)
                # if current_time>=7200 and log_dict['cur_trial']>=20:
                # if current_time>=14000 :#and log_dict['cur_trial']>=14
                #     with open(trial_num_path, 'rb') as f:#input,bug type,params
                #         num = pickle.load(f)
                #     num+=1
                #     with open(trial_num_path, 'wb') as f:
                #         pickle.dump(num, f)
                    
                #     new_dir='{}_{}'.format(root_path,num)
                #     os.rename(root_path,new_dir)
                #     print('finish')
                #     os._exit(0)

                write_algw(new_dir_path)

            return history
        else:
            del fit_kwargs['sub_dataset']
            del fit_kwargs['sub_validation_data']
            
            # with open("/data1/zxy/DL_autokeras/1Autokeras/FORM/FORM/Test_dir/demo_result_twophase/error.pkl", 'rb') as f:#input,bug type,params
            #     error_hps = pickle.load(f)
            # trial.hyperparameters=error_hps
            model = self.hypermodel.build(trial.hyperparameters)
            self.oracle.update_space(trial.hyperparameters)
            # model2 = self.hypermodel.build(trial.hyperparameters)
            
            self.adapt(model, fit_kwargs["x"])


            read_path=os.path.abspath('./Test_dir/tmp/tmp_action_value.pkl')
            if os.path.exists(read_path):
                special=True
                model=read_opt(model)# check for special actions like activation and initializer
            else:
                special=False

            save_path=os.path.abspath('./Test_dir/tmp/gradient_weight.pkl')
            predictor=LossHistory(training_data=data,model=model,total_epoch=fit_kwargs['epochs'],batch_size=batch_size,save_path=save_path)
            fit_kwargs['callbacks'].append(predictor)

            

            _, history = utils.fit_with_adaptive_batch_size(
                model, self.hypermodel.hypermodel.batch_size, **fit_kwargs
            )

            train_history=history.history
            max_val_acc=max(train_history['val_accuracy'])
            trial_num_path=os.path.join(os.path.dirname(root_path),'num.pkl')
            if not os.path.exists(root_path):
                os.makedirs(root_path)

            current_time=time.time()-log_dict['start_time']

            new_dir_name=str(log_dict['cur_trial'])+'-'+str(round(max_val_acc,2))+str(uuid.uuid3(uuid.NAMESPACE_DNS,str(time.time())))[-13:]

            new_dir_path=os.path.join(root_path,new_dir_name)
            
            log_dict[new_dir_name]={}
            log_dict[new_dir_name]['time']=current_time
            log_dict[new_dir_name]['history']=train_history
            log_dict[new_dir_name]['score']=max_val_acc
            log_dict[new_dir_name]['values']=trial.hyperparameters.values

            with open(log_path, 'wb') as f:
                pickle.dump(log_dict, f)

            # TODO: not use in beam search
            trial_id_path=os.path.join(root_path,'trial_id.pkl')
            if os.path.exists(trial_id_path):
                with open(trial_id_path, 'rb') as f:#input,bug type,params
                    trial_id_dict = pickle.load(f)
            else:
                trial_id_dict={}
            if isinstance(trial_id_dict,dict):
                trial_id_dict[trial.trial_id]=new_dir_path
                with open(trial_id_path, 'wb') as f:
                    pickle.dump(trial_id_dict, f)   


            if not os.path.exists(new_dir_path):
                os.makedirs(new_dir_path)
            else:
                new_dir_name=str(log_dict['cur_trial'])+'-'+str(round(max_val_acc,2))+str(uuid.uuid3(uuid.NAMESPACE_DNS,str(time.time())))[-13:]
                new_dir_path=os.path.join(root_path,new_dir_name)
                os.makedirs(new_dir_path)
            
            model_path=os.path.join(new_dir_path,'model.h5')
            history_path=os.path.join(new_dir_path,'history.pkl')
            hyperparam_path=os.path.join(new_dir_path,'param.pkl')
            message_path=os.path.join(new_dir_path,'msg.pkl')
            hm_path = os.path.join(new_dir_path,'hypermodel.pkl')
            new_save_path=os.path.join(new_dir_path,'gradient_weight.pkl')

            if special:
                new_read_path=os.path.join(new_dir_path,'tmp_action_value.pkl')
                
                shutil.move(read_path,new_read_path)

            shutil.move(save_path,new_save_path)

            with open(hm_path, 'wb') as f:
                pickle.dump(self.hypermodel, f)   
            
            with open(message_path, 'wb') as f:
                pickle.dump(msg, f)

            try:
                model.save(model_path)
            except Exception as e:
                print(e)
                model.save(new_dir_path, save_format='tf')

            with open(history_path, 'wb') as f:
                pickle.dump(train_history, f)
            with open(hyperparam_path, 'wb') as f:
                pickle.dump(trial.hyperparameters, f)
            
            # print(1)
            # if current_time>=7200 and log_dict['cur_trial']>=20:
            # if current_time>=14000 :#and log_dict['cur_trial']>=14
            #     with open(trial_num_path, 'rb') as f:#input,bug type,params
            #         num = pickle.load(f)
            #     num+=1
            #     with open(trial_num_path, 'wb') as f:
            #         pickle.dump(num, f)
                
            #     new_dir='{}_{}'.format(root_path,num)
            #     os.rename(root_path,new_dir)
            #     print('finish')
            #     os._exit(0)

            write_algw(new_dir_path)

            return history
    
    def get_new_kwargs(self,fit_kwargs):
        new_fit_kwargs={}
        for key in fit_kwargs:
            if key=='x' or key=='validation_data':
                continue
            elif key== 'sub_dataset':
                new_fit_kwargs['x']=fit_kwargs['sub_dataset']
            elif key == 'sub_validation_data':
                new_fit_kwargs['validation_data']=fit_kwargs['sub_validation_data']
            else:
                new_fit_kwargs[key]=fit_kwargs[key]
        return new_fit_kwargs


    @staticmethod
    def adapt(model, dataset):
        """Adapt the preprocessing layers in the model."""
        # Currently, only support using the original dataset to adapt all the
        # preprocessing layers before the first non-preprocessing layer.
        # TODO: Use PreprocessingStage for preprocessing layers adapt.
        # TODO: Use Keras Tuner for preprocessing layers adapt.
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

    def search(self, epochs=None, callbacks=None, validation_split=0, **fit_kwargs):
        """Search for the best HyperParameters.

        If there is not early-stopping in the callbacks, the early-stopping callback
        is injected to accelerate the search process. At the end of the search, the
        best model will be fully trained with the specified number of epochs.

        # Arguments
            callbacks: A list of callback functions. Defaults to None.
            validation_split: Float.
        """
        if self._finished:
            return

        if callbacks is None:
            callbacks = []

        self.hypermodel.hypermodel.set_fit_args(validation_split, epochs=epochs)

        # Insert early-stopping for adaptive number of epochs.
        epochs_provided = True
        if epochs is None:
            epochs_provided = False
            epochs = 1000
            if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
                callbacks.append(
                    tf_callbacks.EarlyStopping(patience=10, min_delta=1e-4)
                )

        # Insert early-stopping for acceleration.
        early_stopping_inserted = False
        new_callbacks = self._deepcopy_callbacks(callbacks)
        if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
            early_stopping_inserted = True
            new_callbacks.append(
                tf_callbacks.EarlyStopping(patience=10, min_delta=1e-4)
            )

        # Populate initial search space.
        hp = self.oracle.get_space()
        self._prepare_model_build(hp, **fit_kwargs)
        self.hypermodel.build(hp)
        self.oracle.update_space(hp)

        super().search(epochs=epochs, callbacks=new_callbacks, **fit_kwargs)

        # Train the best model use validation data.
        # Train the best model with enough number of epochs.
        if validation_split > 0 or early_stopping_inserted:
            copied_fit_kwargs = copy.copy(fit_kwargs)

            # Remove early-stopping since no validation data.
            # Remove early-stopping since it is inserted.
            copied_fit_kwargs["callbacks"] = self._remove_early_stopping(callbacks)

            # Decide the number of epochs.
            copied_fit_kwargs["epochs"] = epochs
            if not epochs_provided:
                copied_fit_kwargs["epochs"] = self._get_best_trial_epochs()

            # Concatenate training and validation data.
            if validation_split > 0:
                copied_fit_kwargs["x"] = copied_fit_kwargs["x"].concatenate(
                    fit_kwargs["validation_data"]
                )
                copied_fit_kwargs.pop("validation_data")

            self.hypermodel.hypermodel.set_fit_args(
                0, epochs=copied_fit_kwargs["epochs"]
            )
            pipeline, model = self.final_fit(**copied_fit_kwargs)
        else:
            model = self.get_best_models()[0]
            pipeline = pipeline_module.load_pipeline(
                self._pipeline_path(self.oracle.get_best_trials(1)[0].trial_id)
            )

        model.save(self.best_model_path)
        pipeline.save(self.best_pipeline_path)
        self._finished = True

    def get_state(self):
        state = super().get_state()
        state.update({"finished": self._finished})
        return state

    def set_state(self, state):
        super().set_state(state)
        self._finished = state.get("finished")

    @staticmethod
    def _remove_early_stopping(callbacks):
        return [
            copy.deepcopy(callbacks)
            for callback in callbacks
            if not isinstance(callback, tf_callbacks.EarlyStopping)
        ]

    def _get_best_trial_epochs(self):
        best_trial = self.oracle.get_best_trials(1)[0]
        # steps counts from 0, so epochs = step + 1.
        return self.oracle.get_trial(best_trial.trial_id).best_step + 1

    def _build_best_model(self):
        best_trial = self.oracle.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters
        return self.hypermodel.build(best_hp)

    def final_fit(self, **kwargs):
        best_trial = self.oracle.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters
        pipeline, kwargs["x"], kwargs["validation_data"] = self._prepare_model_build(
            best_hp, **kwargs
        )

        model = self._build_best_model()
        self.adapt(model, kwargs["x"])
        model, _ = utils.fit_with_adaptive_batch_size(
            model, self.hypermodel.hypermodel.batch_size, **kwargs
        )
        return pipeline, model

    @property
    def best_model_path(self):
        return os.path.join(self.project_dir, "best_model")

    @property
    def best_pipeline_path(self):
        return os.path.join(self.project_dir, "best_pipeline")

    @property
    def objective(self):
        return self.oracle.objective

    @property
    def max_trials(self):
        return self.oracle.max_trials



def read_data(dataset,batch_size):
    # read data from a new unzipped dataset.
    trainX=dataset['x'][:batch_size]
    trainy=dataset['y'][:batch_size]
    testX=dataset['x_val'][:batch_size]
    testy=dataset['y_val'][:batch_size]
    return trainX,trainy,testX,testy

def extract_dataset(data_x,data_val,method='mnist'):
    tmp_path=os.path.abspath('./Test_dir/tmp/{}.pkl'.format(method))
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
        batch_size=data_x[0][0].shape[0]
        # if data_format==True:
        i = data_x[0]
        try:
            _=i[0].shape[1]
            tmp_i=i[0]
        except:
            tmp_i=i[0].reshape((-1,1))
        
        if dataset['x']==[]:
            dataset['x']=tmp_i
        if dataset['y']==[]:
            dataset['y']=i[1]          
        dataset['batch']=batch_size
        with open(tmp_path, 'wb') as f:
            pickle.dump(dataset, f)
        del dataset['batch']
    return dataset,batch_size