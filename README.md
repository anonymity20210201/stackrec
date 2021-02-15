# StackRec: Efficient Training of Very Deep Sequential Recommender Models by Iterative Stacking

## Datasets
You can download datasets that have been pre-processed:
- ML20: https://pan.baidu.com/s/14pk0N-yraoxGgsnbJRPG5Q code(提取码): 7yha
- ColdRec:
https://pan.baidu.com/s/1AkTImhvnD8WyXCTOuynZ8g code(提取码): 9cs2
https://pan.baidu.com/s/1byW5uCZbdEjGzoXJAlPalQ code(提取码): 856z
- Kuaibao:
https://pan.baidu.com/s/1-_1fY7iSLMgc8jYnqkdcnQ code(提取码): em3g


## Train in the CL scenario

Execute example:

```
sh train_next_sc1.sh
```


## Train in the TS scenario

Execute example:

```
sh train_next_sc2.sh
```

## Train in the TF scenario

Execute example:

```
sh train_next_sc3.sh
```

## Stacking with Transformer architecture

Execute example:

```
sh train_trans_sc1.sh
```


## Key Configuration
- method: five stacking methods including from_scratch, stackC, stackA, stackR and stackE
- data_ratio: the percentage of training data
- dilation_count: the number of  dilation factors {1,2,4,8}
- num_blocks: the number of residual blocks
- load_model: whether load pre-trained model or not
