## listops
# canonicals
python3 run_tasks_time_memory.py --model softmax --task listops 

python3 run_tasks_time_memory.py --model nystrom-64 --task listops 

python3 run_tasks_time_memory.py --model linformer-256 --task listops 

python3 run_tasks_time_memory.py --model performer-256 --task listops 

python3 run_tasks_time_memory.py --model reformer-2 --task listops 

# our PrimalFormer
python3 run_tasks_time_memory.py --model primal_cos --task listops --low_rank 20 --eta 0.05 --trace_no_x


## image
python3 run_tasks_time_memory.py --model softmax --task image

python3 run_tasks_time_memory.py --model nystrom-64 --task image

python3 run_tasks_time_memory.py --model linformer-256 --task image

python3 run_tasks_time_memory.py --model performer-256 --task image

python3 run_tasks_time_memory.py --model reformer-2 --task image

# our PrimalFormer
python3 run_tasks_time_memory.py --model primal_cos --task image --low_rank 20 --eta 0.05 --trace_no_x



## pathfinder
python3 run_tasks_time_memory.py --model softmax --task pathfinder32-curv_contour_length_14

python3 run_tasks_time_memory.py --model nystrom-64 --task pathfinder32-curv_contour_length_14

python3 run_tasks_time_memory.py --model linformer-256 --task pathfinder32-curv_contour_length_14

python3 run_tasks_time_memory.py --model performer-256 --task pathfinder32-curv_contour_length_14

python3 run_tasks_time_memory.py --model reformer-2 --task pathfinder32-curv_contour_length_14

# our PrimalFormer
python3 run_tasks_time_memory.py --model primal_cos --task pathfinder32-curv_contour_length_14 --low_rank 30 --eta 0.1 --trace_no_x


## rerieval
python3 run_tasks_time_memory.py --model softmax --task retrieval

python3 run_tasks_time_memory.py --model nystrom-64 --task retrieval

python3 run_tasks_time_memory.py --model linformer-256 --task retrieval

python3 run_tasks_time_memory.py --model performer-256 --task retrieval

python3 run_tasks_time_memory.py --model reformer-2 --task retrieval

# our PrimalFormer
python3 run_tasks_time_memory.py --model primal_cos --task retrieval --low_rank 30 --eta 0.05 --trace_no_x



## text
python3 run_tasks_time_memory.py --model softmax --task text

python3 run_tasks_time_memory.py --model nystrom-64 --task text

python3 run_tasks_time_memory.py --model linformer-256 --task text

python3 run_tasks_time_memory.py --model performer-256 --task text

python3 run_tasks_time_memory.py --model reformer-2 --task text

# our PrimalFormer
python3 run_tasks_time_memory.py --model primal_cos --task text --low_rank 20 --eta 0.01 --trace_no_x
