#!/bin/bash
source ../grand_env/bin/activate

# The arguments are saved under $1, $2, $3 depending on the order in the command

echo "algo: $1"
echo "num_layers_gnn: $2"
echo "num_layers_dense: $3"
echo "embed_size: $4"

echo "Model name: $5"
echo "Cuda and output: $6"
echo "dataset root: $7"
echo "Topk ratio: $8"

for distrib_infill in 0 5 10 15 20 25 30 35 40
do
   for distrib_coarse in 0 5 10 15 20 25 30 35 40
   do
   nohup python -u core/main.py  --algo=$1  --infill_ratio_test=$distrib_infill --infill_ratio_train=$distrib_infill --coarse_ratio_test=$distrib_coarse --coarse_ratio_train=$distrib_coarse --batch_size=50 --seed=0 --loss_fn=scaled_l1 --root=$7 --embed_size=$4 --lr=1e-3 --num_layers_gnn=$2 --num_layers_dense=$3 --topkratio=$8 --model_name=$5 --device=cuda:$6 --fig_dir_name=${5}_${distrib_infill}_${distrib_coarse} --not_drop_nodes --test > output$6.txt &
   done
done

