model_name=DeepMA

# ---- Monash training subset ----
# Weather is 10-min, 21 meteorological channels; pick weather /
# air-quality / solar series with strong daily seasonality.
monash_dir=/root/monash_tsf_all/data
monash_files=(
  oikolab_weather_dataset.tsf
  weather_dataset.tsf
  temperature_rain_dataset_with_missing_values.tsf
  solar_10_minutes_dataset.tsf
  kdd_cup_2018_dataset_with_missing_values.tsf
)

# ---- Benchmark CSV to evaluate on ----
test_root_path=/root/dataset/weather/
test_data_path=weather.csv

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
