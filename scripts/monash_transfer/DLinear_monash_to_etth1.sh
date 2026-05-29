model_name=DLinear

# ---- Monash training config ----
monash_dir=/root/monash_tsf_all/data
monash_preset=etth_like

# ---- Target benchmark CSV ----
target_root_path=/root/dataset/ETT-small/
target_data_path=ETTh1.csv

seq_len=512

for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name monash_transfer \
  --is_training 1 \
  --root_path $monash_dir \
  --data Monash \
  --model_id Monash_to_ETTh1_${seq_len}_${pred_len} \
  --model $model_name \
  --target_data ETTh1 \
  --target_root_path $target_root_path \
  --target_data_path $target_data_path \
  --monash_preset $monash_preset \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --train_epochs 10 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1
done
