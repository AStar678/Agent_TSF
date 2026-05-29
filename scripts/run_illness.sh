model_name=DeepMA

train_root_path=/root/dataset/illness/
train_data_path=national_illness.csv

test_root_path=/root/dataset/illness/
test_data_path=national_illness.csv

seq_len=36

for pred_len in 24 36 48 60
do
python -u benchmark.py \
  --model ${model_name,,} \
  --train_csv $train_root_path$train_data_path \
  --test_csv $test_root_path$test_data_path \
  --L_in $seq_len \
  --L_out $pred_len
done
