model_name=Linear

train_root_path=/root/dataset/ETT-small/
train_data_path=ETTh1.csv

test_root_path=/root/dataset/ETT-small/
test_data_path=ETTh1.csv

seq_len=1024

for pred_len in 96 192 336 720
do
python -u benchmark.py \
  --model ${model_name,,} \
  --train_csv $train_root_path$train_data_path \
  --test_csv $test_root_path$test_data_path \
  --L_in $seq_len \
  --L_out $pred_len
done
