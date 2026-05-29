model_name=DeepMA

train_root_path=/root/dataset/electricity/
train_data_path=electricity.csv

test_root_path=/root/dataset/ETT-small/
test_data_path=ETTh2.csv

seq_len=720
max_train=500000

for pred_len in 96 192 336 720
do
python -u benchmark.py \
  --model ${model_name,,} \
  --train_csv $train_root_path$train_data_path \
  --test_csv $test_root_path$test_data_path \
  --L_in $seq_len \
  --L_out $pred_len \
  --max_train $max_train
done
