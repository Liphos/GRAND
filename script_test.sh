#!/bin/bash
source ../grand_env/bin/activate

# The arguments are saved under $1, $2, $3 depending on the order in the command

echo "algo: $1"
echo "num_layers: $2"
echo "embed_size: $3"

echo "Model name: $4"
echo "Cuda and output: $5"
echo "dataset root: $6"

for distrib in 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do
   nohup python -u core/main.py --batch_size=50 --seed=0 --loss_fn=scaled_l1 --root=$6 --embed_size=$3 --lr=1e-3 --num_layers=$2 --algo=$1 --model_name=$4 --device=cuda:$5 --ant_ratio_test=$distrib --fig_dir_name=$4$distrib --test > output$5.txt &
   sleep 10
done

