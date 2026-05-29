model_name=DeepMA

train_root_path=/root/dataset/exchange_rate/
train_data_path=exchange_rate.csv

test_root_path=/root/dataset/exchange_rate/
test_data_path=exchange_rate.csv

seq_len=96

for pred_len in 96 192 336 720
do
python -u benchmark.py \
  --model ${model_name,,} \
  --train_csv $train_root_path$train_data_path \
  --test_csv $test_root_path$test_data_path \
  --L_in $seq_len \
  --L_out $pred_len
done
