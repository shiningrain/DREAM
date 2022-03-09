import os
import sys
sys.path.append('.')
import copy
import numpy as np
import keras
import datetime
# from keras.models import load_model,Sequential
import keras.backend as K
import tensorflow as tf
import time
import uuid
from collections import Counter
import copy
import csv
import pickle
from autokeras.utils import opt
from autokeras.utils import autotrainer

# from keras.models import load_model
# from keras.models import Model
# from keras.activations import relu,sigmoid,elu,linear,selu
# from keras.layers import BatchNormalization,GaussianNoise
# from keras.layers import Activation,Add,Dense
# from keras.layers.core import Lambda
# from keras.initializers import he_uniform,glorot_uniform,zeros
# from keras.optimizers import SGD, Adam, Adamax
# from keras.callbacks import ReduceLROnPlateau
temp_fix_saved_dir='/data/zxy/DL_autokeras/1Autokeras/test_codes/experiment/autokeras_log/tmp_fix_saved_dir'


# solution_evaluation = {
#         'gradient' : {'modify':0, 'memory':False},
#         'relu': {'modify':1, 'memory':False},
#         'bn': {'modify':2, 'memory':True},
#         'initial': {'modify':1, 'memory':False},
#         'selu': {'modify':1 , 'memory':False},
#         'tanh': {'modify':1 , 'memory':False},
#         'leaky': {'modify':2 , 'memory':False},
#         'adam' : {'modify':0 , 'memory':False},
#         'lr': {'modify':0 , 'memory':False},
#         'ReduceLR': {'modify':0 , 'memory':False},
#         'momentum' : {'modify':0 , 'memory':False},
#         'batch': {'modify':0 , 'memory':False},
#         'GN': {'modify':2 , 'memory':False},
#         'optimizer': {'modify':0 , 'memory':False},
#         'regular':{'modify':1 , 'memory':False},#how
#         'dropout':{'modify':2 , 'memory':False},#how
#         'estop':{'modify':0, 'memory':False}
#     }
problem_evaluation = {# please use the priority order here
    'vanish': 2,
    'explode': 2,
    'relu': 2,
    'not_converge': 1,
    'unstable': 1,
    'overfit': 0,
    'need_train':0
}

def csv_determine(issue_name,path):
    file_name=issue_name+'.csv'
    file_path=os.path.join(path,file_name)
    return file_path

def filtered_issue(issue_list):
    if len(issue_list)>1:
        new_issue_list=[]
        for i in range(len(issue_list)):
            if new_issue_list==[] or problem_evaluation[issue_list[i]]>problem_evaluation[new_issue_list[0]]:
                new_issue_list=[issue_list[i]]
        return new_issue_list
    else:return issue_list

def comband_solutions(solution_dic):
    solution=[]
    stop=False
    i=0
    while(stop==False):
        solution_start=len(solution)
        for key in solution_dic:
            if i <= (len(solution_dic[key])-1):
                solution.append(solution_dic[key][i])
        if solution_start==len(solution): stop=True
    return solution


def filtered(strategy_list):
    return strategy_list[0],strategy_list[1],strategy_list[2],strategy_list[3],strategy_list[4],strategy_list[5]

def get_each_strategy(strategy_list):
    for strat in range(len(strategy_list)):
        for i in range(len(strategy_list[strat])):
            sol=strategy_list[strat][i].split('_')[0]
    return strategy_list

def merge_history(history,new_history):
    if history=={}:
        history=new_history.copy()
        history['train_node']=[len(new_history['loss'])]
        return history
    for i in history.keys():
        if i in new_history.keys():
            for j in range(len(new_history[i])):
                history[i].append(new_history[i][j])
    history['train_node'].append(len(new_history['loss']))
    return history

def read_strategy(string):
    solution=string.split('_')[0]
    times= string.split('_')[-1]
    #if string.split('_')[-1]=='': times=1
    return solution,int(times)


def get_new_dir(new_issue_dir,case_name,issue_type,tmp_add):
    case_name=case_name.split('/')[-1]
    new_case_name=case_name+'-'+tmp_add
    new_issue_type_dir=os.path.join(new_issue_dir,issue_type)
    new_case_dir=os.path.join(new_issue_type_dir,new_case_name)
    if not os.path.exists(new_case_dir):
        os.makedirs(new_case_dir)
    return new_case_dir

def notify_result(num, model,train_config,issue,j):
    #(tmp_sol,model,issue_type,j)
    numbers = {
        'gradient' : opt.op_gradient,
        'relu': opt.op_relu,
        'bn': opt.op_bn,
        'initial': opt.op_initial,
        'selu': opt.op_selu,
        'leaky': opt.op_leaky,
        'adam' : opt.op_adam,
        'lr': opt.op_lr,
        # 'ReduceLR': opt.op_ReduceLR,
        'momentum' : opt.op_momentum,
        'batch': opt.op_batch,
        'GN': opt.op_GN,
        'optimizer': opt.op_optimizer,
        # 'regular':opt.op_regular,
        # 'dropout':opt.op_dropout,
        # 'estop':opt.op_EarlyStop,
        'tanh':opt.op_tanh
    }

    method = numbers.get(num, opt.repair_default)
    if method:
        return method(model,train_config,issue,j)

class Repair_Module:
    def __init__(self, model, train_config,issue_list,satisfied_acc, checktype='epoch_1', determine_threshold=5):
        """#pure_config,
        method:['efficiency','structure','balance'], efficient will try the most efficiently solution and the structure will
            first consider to keep the model structure/training configuration.balance is the compromise solution.
        """
        self.satisfied_acc = satisfied_acc
        self.model = model
        self.issue_list = issue_list
        # self.root_path=root_path
        # if not os.path.exists(root_path):
        #     os.makedirs(root_path)

        self.initial_issue = copy.deepcopy(issue_list)
        self.train_config = train_config
        self.config_bk = train_config.copy()
        self.best_potential = []
        self.checktype = checktype
        self.determine_threshold = determine_threshold

        strategy_list = opt.repair_strategy()
        [self.gradient_vanish_strategy, self.gradient_explode_strategy, self.dying_relu_strategy,\
            self.unstable_strategy, self.not_converge_strategy]=strategy_list


    def solve(self): 
        # 
        # This version not try for the new problems,just fix current
        solved_issue=[]
        history={}
        # tmp_add=''
        for i in range(3):#10
        #case_name=tmp_dir.replace('/monitor_tool_log/solution','')
            case_name=str(uuid.uuid4()).split('-')[0]
            # get a short string as current model name

            result, result_list= self.issue_repair(
                self.model,
                self.config_bk,
                self.issue_list,
                case_name=case_name)

            # result_list=[new_model,retrain_history,train_result,config,new_issue_list,modify]
            # if problem is totally fixed, the element "new_issue_list" will be removed 

            train_history=result_list[1]
            solved_issue.append(self.issue_list[0])

            if result == 'solved':
                print('Your model has been trained and the training problems {} have been repaired.'.format(self.initial_issue))
                return result,result_list[0],result_list[1],solved_issue,result_list[4],result_list[3],result_list[5]

                # result,             model,        history,  issue_list, now_issue
            elif result == 'no' and i==0:
                # [new_model,retrain_history,train_result,config,new_issue_list]
                print(
                    'Your model still has training problems {} are still exist'
                    .format(result_list[4]))
                return result,result_list[0],result_list[1],solved_issue,result_list[4],result_list[3],result_list[5]
            else:
                if result=='no':
                    self.best_potential[0].save(model_path)
                    print('Model has been improved but still has problem. The initial training problems in the source model is {}.\
                        The current problem is {}.'.format(self.initial_issue,self.best_potential[4]))
                    return result,self.best_potential[0],self.best_potential[1],solved_issue,self.best_potential[4],self.best_potential[3],self.best_potential[5]

                # [new_model,retrain_history,train_result,config,new_issue_list]
                potential_list=result_list
                self.issue_list=potential_list[4]
                self.model=potential_list[0]
                self.train_config=potential_list[3]
                    
                if self.issue_list[0] not in solved_issue:    
                    #save new model
                    model_path=os.path.join(temp_fix_saved_dir,'new_model.h5')
                    
                    if self.best_potential==[] or self.best_potential[2]<potential_list[2]:
                        self.best_potential=potential_list.copy()

                    history=train_history
                    del potential_list
                else:
                    print('Model has been improved but still has problem. The initial training problems in the source model is {}.\
                        The current problem is {}. You can find the best improved model is saved \
                        in {}.'.format(self.initial_issue,self.best_potential[4],model_path))
                    return result,self.best_potential[0],self.best_potential[1],solved_issue,self.best_potential[4],self.best_potential[3],self.best_potential[5]


    def issue_repair(self,seed_model,config_bk,issue_list,case_name):#need modify tmp_add=''
        """[summary]
        result, result_list,new_config_set= self.issue_repair( 
            self.model,
            self.issue_list,
            case_name=case_name)
        Args:
            seed_model ([type]): [description]
            train_config ([type]): [description]
            config_bk ([type]): [description]
            tmp_dir ([type]): [description]
            issue_list (bool): [description]
            tmp_add (str, optional): [description]. Defaults to ''.
            max_try (int, optional): [Max try solution, if not solve in this solution and has potential, then try to solve the potential]. Defaults to 2.

        Returns:
            [type]: [description]
        """
        issue_type=issue_list[0]
        issue_name,solution_list=self.get_file_name(issue_type)
        start_time=time.time()
        potential=[]
        log=[]
        length_solution=len(solution_list)
        try_count=0

        self.seed_model_path='/data/zxy/DL_autokeras/1Autokeras/test_codes/experiment/tmp_model/seed_model.h5'
        if os.path.isfile(self.seed_model_path):
            os.remove(self.seed_model_path)# need the before backup model delete it later
        try:
            seed_model.save(self.seed_model_path)# save the seed model as a backup.
        except Exception as e:
            if os.path.isfile(self.seed_model_path):
                print('Get model')
                # pass
            else:
                seed_model.save(self.seed_model_path,save_format='tf')



        for i in range(3):#length_solution

            tmp_sol,tmp_tim=read_strategy(solution_list[i])
            self.update_current_solution(issue_type,solution_list[i])
            for j in range(tmp_tim):
                # solutions
                _break=False

                
                from keras.models import load_model
                # model=load_model('/data/zxy/DL_autokeras/1Autokeras/test_codes/relu/b5e3c487/model.h5')# linshi
                model = load_model(self.seed_model_path)
                # try:
                #     model=copy.deepcopy(seed_model)
                    
                # except:
                #     model=model#seed_model
                #     print('Warning: Model failed in copy.')
                train_config=copy.deepcopy(config_bk)

                tmp_model,config,modify,_break=notify_result(tmp_sol,model,train_config,issue_type,j)
                # tmp_model=load_model('/data/zxy/DL_autokeras/1Autokeras/test_codes/relu/b5e3c487/tmp_fix.h5')
                # config=train_config

                if _break: break#  the solution has already been used in the source model.
                print('-------------Solution {} has been used, waiting for retrain.-----------'.format(tmp_sol))
                             
                new_model,new_issue_list,train_result,retrain_history=autotrainer.model_retrain(
                    tmp_model,
                    config,
                    satisfied_acc=self.satisfied_acc,
                    solution=tmp_sol,
                    determine_threshold=self.determine_threshold,
                    checktype=self.checktype)
                

                if new_issue_list==[]:
                    end_time=time.time()
                    time_used=end_time-start_time
                    print('------------------Solved! Time used {}!-----------------'.format(str(time_used)))
                    return 'solved', [new_model,retrain_history,train_result,config,[],modify]
                
                elif issue_type not in new_issue_list :
                    if(potential==[] or potential[2]<train_result):
                        potential=[new_model,retrain_history,train_result,config,new_issue_list,modify]

                if log==[] or log[2]<train_result:
                    log=[new_model,retrain_history,train_result,config,new_issue_list,modify]
            

        if potential==[]: 
            end_time=time.time()
            time_used=end_time-start_time
            print('------------------Unsolved..Time used {}!-----------------'.format(str(time_used)))
            return 'no',log #[new_model,retrain_history,train_result,config,new_issue_list,modify]

        else:
            end_time=time.time()
            time_used=end_time-start_time
            print('------------------Not totally solved..Time used {}!-----------------'.format(str(time_used)))
            return 'potential',potential # [new_model,retrain_history,train_result,config,new_issue_list,modify]

    def get_file_name(self,issue_type):
        if issue_type=='vanish':
            return 'gradient_vanish',self.gradient_vanish_strategy.copy()
        elif issue_type=='explode':
            return 'gradient_explode',self.gradient_explode_strategy.copy()
        elif issue_type=='relu':
            return 'dying_relu',self.dying_relu_strategy.copy()
        elif issue_type=='unstable':
            return 'training_unstable',self.unstable_strategy.copy()
        elif issue_type=='not_converge':
            return 'training_not_converge',self.not_converge_strategy.copy()
        elif issue_type=='overfit':
            return 'over_fitting',self.over_fitting_strategy.copy()
    
    def update_current_solution(self,issue_type,solution):
        if issue_type=='vanish':
            return self.gradient_vanish_strategy.remove(solution)
        elif issue_type=='explode':
            return self.gradient_explode_strategy.remove(solution)
        elif issue_type=='relu':
            return self.dying_relu_strategy.remove(solution)
        elif issue_type=='unstable':
            return self.unstable_strategy.remove(solution)
        elif issue_type=='not_converge':
            return self.not_converge_strategy.remove(solution)
        elif issue_type=='overfit':
            return self.over_fitting_strategy.remove(solution)

