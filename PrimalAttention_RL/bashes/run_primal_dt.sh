## Medium-Expert
# HalfCheetah
python experiment.py --env halfcheetah --dataset medium-expert --work_dir primalformer --n_head 4 --n_layer 3 --embed_dim 256 --learning_rate 0.001 --eta 0.05 --low_rank 96

# Hopper
python experiment.py --env hopper --dataset medium-expert --work_dir primalformer --n_head 4 --n_layer 3 --embed_dim 256 --learning_rate 0.001 --eta 0.05 --low_rank 64

# Walker
python experiment.py --env walker2d --dataset medium-expert --work_dir primalformer --n_head 4 --n_layer 3 --embed_dim 256 --learning_rate 0.001 --eta 0.05 --low_rank 96


## Medium
# HalfCheetah
python experiment.py --env halfcheetah --dataset medium --work_dir primalformer --n_head 4 --n_layer 3 --embed_dim 256 --learning_rate 0.001 --eta 0.05 --low_rank 64

# Hopper
python experiment.py --env halfcheetah --dataset medium --work_dir primalformer --n_head 4 --n_layer 3 --embed_dim 256 --learning_rate 0.001 --eta 0.05 --low_rank 96

# Walker
python experiment.py --env walker2d --dataset medium --work_dir primalformer --n_head 4 --n_layer 3 --embed_dim 256 --learning_rate 0.001 --eta 0.05 --low_rank 96


## Medium-Replay
# HalfCheetah
python experiment.py --env halfcheetah --dataset medium-replay --work_dir primalformer --n_head 4 --n_layer 3 --embed_dim 256 --learning_rate 0.001 --eta 0.05 --low_rank 32

# Hopper
python experiment.py --env hopper --dataset medium-replay --work_dir primalformer --n_head 4 --n_layer 3 --embed_dim 256 --learning_rate 0.001 --eta 0.05 --low_rank 64

# Walker
python experiment.py --env walker2d --dataset medium-replay --work_dir primalformer --n_head 4 --n_layer 3 --embed_dim 256 --learning_rate 0.001 --eta 0.05 --low_rank 96
