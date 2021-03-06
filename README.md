# StackRec: Efficient Training of Very Deep Sequential Recommender Models by Iterative Stacking

## Datasets
You can download datasets that have been pre-processed:
- ML20: https://pan.baidu.com/s/14pk0N-yraoxGgsnbJRPG5Q code(提取码): 7yha
- ColdRec:
https://pan.baidu.com/s/1AkTImhvnD8WyXCTOuynZ8g code(提取码): 9cs2
https://pan.baidu.com/s/1byW5uCZbdEjGzoXJAlPalQ code(提取码): 856z
- Kuaibao: Permission is required from Tencent. We will open source after paper acceptance.


## File Description
```
requirements.txt: the experiment environment

train_nextitnet_sc1.sh: the shell script to train StackRec with NextItNet in CL scenario
train_nextitnet_sc2.sh: the shell script to train StackRec with NextItNet in TF scenario
train_nextitnet_sc3.sh: the shell script to train StackRec with NextItNet in TS scenario
deep_nextitnet.py: the training file of NextItNet
deep_nextitnet_coldrec.py: the training file of NextItNet customized for coldrec source dataset
data_loader.py: the dataset loading file of NextItNet and GRec
data_loader_finetune.py: the dataset loading file of NextItNet and GRec customized for coldrec dataset
generator_deep.py: the model file of NextItNet
ops.py: the module file of NextItNet and GRec with stacking methods doubling blocks
ops_copytop.py: the module file of NextItNet with stacking methods allowed to stack top blocks
ops_original.py: the module file of NextItNet with stacking methods without alpha
fineall.py: the training file of NextItNet customized for coldrec target dataset

train_grec_sc1.sh: the shell script to train StackRec with GRec in CL scenario
deep_GRec: the training file of GRec
generator_deep_GRec.py: the model file of GRec
utils_GRec.py: some tools for GRec

train_sasrec_sc1.sh: the shell script to train StackRec with SASRec in CL scenario
baseline_SASRec.py: the training file of SASRec
Data_loader_SASRec.py: the dataset loading file of SASRec
SASRec_Rezero.py: the model file of SASRec

train_ssept_sc1.sh: the shell script to train StackRec with SSEPT in CL scenario
baseline_SSEPT.py: the training file of SSEPT
Data_loader_SSEPT.py: the dataset loading file of SSEPT
SSEPT_Rezero.py: the model file of SSEPT
utils.py: some tools for SASRec and SSEPT
Modules.py: the module file of SASRec and SSEPT with stacking methods
```

## Stacking with NextItNet
### Train in the CL scenario

Execute example:

```
sh train_nextitnet_sc1.sh
```

### Train in the TS scenario

Execute example:

```
sh train_nextitnet_sc2.sh
```

### Train in the TF scenario

Execute example:

```
sh train_nextitnet_sc3.sh
```

## Stacking with GRec

Execute example:

```
sh train_grec_sc1.sh
```

## Stacking with SASRec

Execute example:

```
sh train_sasrec_sc1.sh
```

## Stacking with SSEPT

Execute example:

```
sh train_ssept_sc1.sh
```

## Key Configuration
- method: five stacking methods including from_scratch, stackC, stackA, stackR and stackE
- data_ratio: the percentage of training data
- dilation_count: the number of  dilation factors {1,2,4,8}
- num_blocks: the number of residual blocks
- load_model: whether load pre-trained model or not
