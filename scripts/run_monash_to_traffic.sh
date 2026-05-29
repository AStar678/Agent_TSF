model_name=DeepMA

# ---- Monash training subset ----
# Traffic is hourly road occupancy (862 sensors); pick traffic flow +
# pedestrian counts + hourly demand series with daily/weekly seasonality.
monash_dir=/root/monash_tsf_all/data
monash_files=(
  pedestrian_counts_dataset.tsf
  rideshare_dataset_with_missing_values.tsf
)

# ---- Benchmark CSV to evaluate on ----
test_root_path=/root/dataset/traffic/
test_data_path=traffic.csv

seq_len=720
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
