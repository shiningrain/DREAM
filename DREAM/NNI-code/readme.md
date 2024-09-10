## NNI experiments

First, you need to use `tpe_tuner.py` to replace `YOUR_ENV/site-packages/nni/algorithms/hpo/tpe_tuner.py`, use `gridsearch_tuner.py` to replace `YOUR_ENV/site-packages/nni/algorithms/hpo/gridsearch_tuner.py`, and use `formatting.py` to replace `YOUR_ENV/site-packages/nni/common/hpo_utils/formatting.py`,

Then, refer to NNI examples, please use `launch.py` to call `c100/reuters/tiny.py` and run experiments.