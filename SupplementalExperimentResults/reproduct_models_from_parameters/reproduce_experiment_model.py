import pickle
import autokeras as ak

with open('./hypermodel.pkl', 'rb') as f:
    hm = pickle.load(f)
with open('./param.pkl', 'rb') as f: #you need to input the parameter of the model here
    param = pickle.load(f)

best_model=hm.build(param) #the model will be build by autokeras with this parameter 
print('finish')