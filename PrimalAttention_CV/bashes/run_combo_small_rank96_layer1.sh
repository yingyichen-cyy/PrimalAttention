# Please change "--data-path" and "--out-dir" to your own paths
# 2 A100. each batch_size = 512
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_combo.py --model primal_small_patch16_224 --batch-size 512 --data-path ./imagenet-100 --data-set IMNET-100 --low-rank 96 --rank-multi 10 --num-ksvd-layer 1 --eta 0.05 --output_dir ./combo_small_rank96_multi10_eta005_numlayer1_results
