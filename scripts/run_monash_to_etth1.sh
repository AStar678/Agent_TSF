model_name=DeepMA

# ---- Monash training subset (edit to pick which .tsf files to pretrain on) ----
monash_dir=/root/monash_tsf_all/data
monash_files=(
  australian_electricity_demand_dataset.tsf
  traffic_hourly_dataset.tsf
  kdd_cup_2018_dataset_with_missing_values.tsf
  oikolab_weather_dataset.tsf
  tourism_quarterly_dataset.tsf
)

# ---- Benchmark CSV to evaluate on ----
test_root_path=/root/dataset/ETT-small/
test_data_path=ETTh1.csv

seq_len=720

for pred_len in 96 192 336 720
do
python -u benchmark.py \
  --model ${model_name,,} \
  --train_source monash \
  --monash_dir $monash_dir \
  --monash_files "${monash_files[@]}" \
  --test_csv $test_root_path$test_data_path \
  --L_in $seq_len \
  --L_out $pred_len
done
