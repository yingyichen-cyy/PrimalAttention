# listops
python3 run_tasks.py --model primal_cos --task listops --low_rank 20 --eta 0.1 --trace_no_x

# image 
python3 run_tasks.py --model primal_cos --task image --low_rank 30 --eta 0.1 --trace_no_x

# pathfinder 
python3 run_tasks.py --model primal_cos --task pathfinder32-curv_contour_length_14 --low_rank 30 --eta 0.1 --trace_no_x

# retrieval
python3 run_tasks.py --model primal_cos --task retrieval --low_rank 30 --eta 0.0 --trace_no_x

# text
python3 run_tasks.py --model primal_cos --task text --low_rank 20 --eta 0.05 --trace_no_x