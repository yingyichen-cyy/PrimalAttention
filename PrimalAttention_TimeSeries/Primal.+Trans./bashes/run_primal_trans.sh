export CUDA_VISIBLE_DEVICES=1

## Primal.+Trans. with feature maps realted to the 
## Cosine similarity kernel on queries and keys

# EthanolConcentration
python main.py --output_dir experiments --comment "classification for Primalformer" --name EthanolConcentration_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/EthanolConcentration --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam --pos_encoding learnable --task classification --key_metric accuracy --model primal_cos --low_rank 40 --eta 0.2 --rank_multi 10

# FaceDetection
python main.py --output_dir experiments --comment "classification for Primalformer" --name FaceDetection_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/FaceDetection --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model primal_cos --low_rank 40 --eta 0.2 --rank_multi 5

# Handwriting
python main.py --output_dir experiments --comment "classification for Primalformer" --name Handwriting_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/Handwriting --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model primal_cos --low_rank 20 --eta 0.5 --rank_multi 5

# HeartBeat
python main.py --output_dir experiments --comment "classification for Primalformer" --name Heartbeat_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/Heartbeat --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model primal_cos --low_rank 40 --eta 0.5 --rank_multi 10

# Japanese
python main.py --output_dir experiments --comment "classification for Primalformer" --name JapaneseVowels_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/JapaneseVowels --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model primal_cos --low_rank 30 --eta 0.2 --rank_multi 5

# PEMS-SF
python main.py --output_dir experiments --comment "classification for Primalformer" --name PEMS-SF_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/PEMS-SF --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 400 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model primal_cos --low_rank 40 --eta 0.0 --rank_multi 5

# SelfRegulationSCP1
python main.py --output_dir experiments --comment "classification for Primalformer" --name SelfRegulationSCP1_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/SelfRegulationSCP1 --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model primal_cos --low_rank 30 --eta 0.2 --rank_multi 10

# SelfRegulationSCP2
python main.py --output_dir experiments --comment "classification for Primalformer" --name SelfRegulationSCP2_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/SelfRegulationSCP2 --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model primal_cos --low_rank 20 --eta 0.2 --rank_multi 10

# SpokenArabicDigits
python main.py --output_dir experiments --comment "classification for Primalformer" --name SpokenArabicDigits_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/SpokenArabicDigits --data_class tsra  --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer Adam  --pos_encoding learnable  --task classification  --key_metric accuracy --model primal_cos --low_rank 30 --eta 0.2 --rank_multi 5

# UWave
python main.py --output_dir experiments --comment "classification for Primalformer" --name UWaveGestureLibrary_fromScratch --records_file Classification_records.xls --data_dir Multivariate_ts/UWaveGestureLibrary --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --model primal_cos --low_rank 40 --eta 0.1 --rank_multi 10
