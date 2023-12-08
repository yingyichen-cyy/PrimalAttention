# Please change "--data-path" and "--out-dir" to your own paths
# 2 A100. each batch_size = 512
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model deit_small_patch16_224 --batch-size 512 --data-path ./imagenet-100 --data-set IMNET-100 --output_dir ./deit_small_results
