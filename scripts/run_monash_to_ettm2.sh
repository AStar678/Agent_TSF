model_name=Linear

# ---- Monash training subset ----
# ETTm2 is 15-min electrical-transformer; same picks as ETTm1.
monash_dir=/root/monash_tsf_all/data
monash_files=(
  london_smart_meters_dataset_with_missing_values.tsf
  australian_electricity_demand_dataset.tsf
  solar_10_minutes_dataset.tsf
  wind_farms_minutely_dataset_with_missing_values.tsf
  oikolab_weather_dataset.tsf
)
# ---- Benchmark CSV to evaluate on ----
test_root_path=/root/dataset/ETT-small/
test_data_path=ETTm2.csv

seq_len=720

for pred_len in 720
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
