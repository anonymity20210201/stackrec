#!/usr/bin/env bash
data_ratio1=0.8
data_ratio2=1

num_blocks1=18
num_blocks2=24

eval_iter1=6000
step1=350000

eval_iter2=6000
step2=350000

early_stop1=8
early_stop2=8

learning_rate1=0.001
learning_rate2=0.001

method1='from_scratch'
method2='stackA'

load_model1=False
load_model2=True

seed=a0205
save_dir="Models/ml20_${num_blocks1}_${num_blocks2}_${seed}"

mkdir -p ${save_dir}

time=$(date "+%Y%m%d%H%M%S")
logfile="${save_dir}/log_${time}.txt"


model_path1=${save_dir}
model_path2="${save_dir}/${num_blocks1}_${learning_rate1}_${data_ratio1}_${step1}.pkl"

gpu=4

eval "CUDA_VISIBLE_DEVICES=${gpu} python -u baseline_SASRec.py --eval_iter ${eval_iter1} --data_ratio ${data_ratio1} --step ${step1} --early_stop ${early_stop1} --learning_rate ${learning_rate1} --num_blocks ${num_blocks1} --method ${method1} --load_model ${load_model1}  --model_path ${model_path1} --save_dir ${save_dir} | tee ${logfile}"

eval "CUDA_VISIBLE_DEVICES=${gpu} python -u baseline_SASRec.py --eval_iter ${eval_iter2} --data_ratio ${data_ratio2} --step ${step2} --early_stop ${early_stop2} --learning_rate ${learning_rate2} --num_blocks ${num_blocks2} --method ${method2} --load_model ${load_model2}  --model_path ${model_path2} --save_dir ${save_dir} | tee -a ${logfile}"

