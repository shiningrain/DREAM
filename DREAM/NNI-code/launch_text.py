# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Example showing how to create experiment with Python code.
"""
import os
from pathlib import Path
import shutil
from nni.experiment import Experiment
import json
import argparse
import time
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test image classification')
    parser.add_argument('--save_dir','-sd',default='./demo_result-reuters', help='searching result path')
    parser.add_argument('--search_space','-sp',default='./search_space-text.json', help='searching result path')
    parser.add_argument('--max_trial','-tr',default=100, type=int, help='searching result path')
    parser.add_argument('--max_time','-tm',default='6h', help='searching result path')
    parser.add_argument('--port','-pt',default=8081, type=int,help='searching result path')
    parser.add_argument('--device','-dv',default=0, type=int,help='searching result path')

    # trial setting
    parser.add_argument('--source_param','-prm',default='./reuters_initial/param_reuters_0-new.pkl', type=str, help='source_param path')
    parser.add_argument('--tmp_dir','-td',default='./tmp', type=str, help='searching result path')
    parser.add_argument('--epoch','-ep',default=200, type=int, help='epoch')
    parser.add_argument('--batch_size','-bs',default=32, type=int, help='batch_size')
    parser.add_argument('--hypermodel_path','-hp',default='./hypermodel-reuters.pkl', type=str, help='hypermodel_path')

    args = parser.parse_args()

    with open(args.search_space, 'r') as f:
        search_space = json.load(f)

    search_space['hyper_path']=os.path.abspath(args.source_param)

    experiment = Experiment('local')
    experiment.config.experiment_name = 'Reuters test 0806'
    experiment.config.trial_concurrency = 1
    experiment.config.search_space = search_space
    experiment.config.trial_code_directory = Path(__file__).parent
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.tuner.class_args['seed']=12345,
    experiment.config.tuner.class_args['tpe_args']={
            'constant_liar_type': 'mean',
            'n_startup_jobs': 10,
            'n_ei_candidates': 20,
            'linear_forgetting': 100,
            'prior_weight': 0,
            'gamma': 0.5
        }
    # experiment.config.tuner.name = 'Random'
    # experiment.config.tuner.class_args = {
    #     'seed': 100
    # }
    experiment.config.trial_gpu_number = 1
    experiment.config.training_service.use_active_gpu = True
    experiment.config.debug=True
    experiment.config.tuner_gpu_indices = 0
    
    experiment.config.experiment_working_directory = args.save_dir
    experiment.config.max_trial_number = args.max_trial
    # experiment.config.max_experiment_duration  = args.max_time
    experiment.config.trial_command = f'CUDA_VISIBLE_DEVICES={args.device} python reuters.py \
        --save_dir {args.save_dir} \
        --tmp_dir {args.tmp_dir} \
        --epoch {args.epoch} \
        --batch_size {args.batch_size} \
        --hypermodel_path {args.hypermodel_path}'
    
    if os.path.exists(experiment.config.experiment_working_directory):
        shutil.rmtree(experiment.config.experiment_working_directory)
    os.makedirs(experiment.config.experiment_working_directory)
    log_path=os.path.join(experiment.config.experiment_working_directory,'log.pkl')
    if not os.path.exists(log_path):
        log_dict={}
        log_dict['cur_trial']=-1
        log_dict['start_time']=time.time()
        log_dict['origin_param_path']=os.path.abspath(args.source_param)

        with open(log_path, 'wb') as f:
            pickle.dump(log_dict, f)
    else:
        os._exit(0)


    experiment.run(args.port)
