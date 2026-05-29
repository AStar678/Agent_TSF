model_name=DeepMA

# ---- Monash training subset ----
# ECL is hourly electricity consumption (321 clients); pick electricity
# demand + smart-meter + hourly traffic/weather series.
monash_dir=/root/monash_tsf_all/data
monash_files=(
  australian_electricity_demand_dataset.tsf
  london_smart_meters_dataset_with_missing_values.tsf
  traffic_hourly_dataset.tsf
  kdd_cup_2018_dataset_with_missing_values.tsf
  solar_10_minutes_dataset.tsf
)

# ---- Benchmark CSV to evaluate on ----
test_root_path=/root/dataset/electricity/
test_data_path=electricity.csv

seq_len=96
max_train=500000

for pred_len in 96 192 336 720
do
python -u benchmark.py \
  --model ${model_name,,} \
  --train_source monash \
  --monash_dir $monash_dir \
  --monash_files "${monash_files[@]}" \
  --test_csv $test_root_path$test_data_path \
  --L_in $seq_len \
  --L_out $pred_len \
  --max_train $max_train
done
