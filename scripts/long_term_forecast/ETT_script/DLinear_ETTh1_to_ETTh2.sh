model_name=DLinear

# Cross-dataset transfer:
#   - Train + validate on ETTh1 (source)
#   - Test on ETTh2 (target) using the checkpoint trained on ETTh1
#
# Implementation: we added --target_data / --target_root_path / --target_data_path
# in run.py. When set, data_provider swaps to the target dataset only for the
# 'test' split. Train/val still use --data (source).
#
# Notes:
# 1. ETTh1 and ETTh2 share the same schema (7 channels, OT target), so enc_in /
#    dec_in / c_out stay 7. For other transfers, channel counts must match.
# 2. Each ETT dataset fits its own StandardScaler on its own train split
#    (see Dataset_ETT_hour). This is the standard transfer-evaluation paradigm.

# ---- 1) Train on ETTh1, then auto-test on ETTh2 ----
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /root/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_to_ETTh2_96_96 \
  --model $model_name \
  --data ETTh1 \
  --target_data ETTh2 \
  --target_root_path /root/dataset/ETT-small/ \
  --target_data_path ETTh2.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'ETTh1_to_ETTh2' \
  --itr 1

# ---- 2) (Optional) Re-run only the test on ETTh2 from the saved checkpoint ----
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 0 \
#   --root_path /root/dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_to_ETTh2_96_96 \
#   --model $model_name \
#   --data ETTh1 \
#   --target_data ETTh2 \
#   --target_root_path /root/dataset/ETT-small/ \
#   --target_data_path ETTh2.csv \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'ETTh1_to_ETTh2' \
#   --itr 1
