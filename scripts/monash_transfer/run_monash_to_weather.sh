model_name=RLinear

# ---- Monash training subset ----
monash_dir=/root/monash_tsf_all/data
monash_files=(
  australian_electricity_demand_dataset.tsf
  traffic_hourly_dataset.tsf
  kdd_cup_2018_dataset_with_missing_values.tsf
  oikolab_weather_dataset.tsf
  pedestrian_counts_dataset.tsf
)

# ---- Benchmark CSV to evaluate on ----
root_path=/root/dataset/
data_path=weather.csv

seq_len=512

for pred_len in 96 192 336 720
do
python -u run_ols.py \
  --model ${model_name,,} \
  --train_source monash \
  --monash_dir $monash_dir \
  --monash_files "${monash_files[@]}" \
  --root_path $root_path \
  --data_path $data_path \
  --seq_len $seq_len \
  --pred_len $pred_len
done
