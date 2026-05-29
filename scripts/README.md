# DeepMA Benchmark Scripts

Per-dataset scripts for training & testing **DeepMA-Linear** under the
**Time-MoE** benchmark protocol (fixed ETTh/ETTm month splits, 70/10/20 for
others; StandardScaler fit on train only; channel-independent).

Each script is a thin wrapper around `benchmark.py` that loops over prediction
lengths. Run from the project root:

```bash
cd /root/DeepMA
bash scripts/run_etth1.sh
```

## Scripts

| Script                 | Dataset                                      | `seq_len` | `pred_len`        |
|------------------------|----------------------------------------------|-----------|-------------------|
| `run_etth1.sh`         | `ETT-small/ETTh1.csv`                        | 96        | 96 192 336 720    |
| `run_etth2.sh`         | `ETT-small/ETTh2.csv`                        | 96        | 96 192 336 720    |
| `run_ettm1.sh`         | `ETT-small/ETTm1.csv`                        | 96        | 96 192 336 720    |
| `run_ettm2.sh`         | `ETT-small/ETTm2.csv`                        | 96        | 96 192 336 720    |
| `run_electricity.sh`   | `electricity/electricity.csv`                | 96        | 96 192 336 720    |
| `run_exchange.sh`      | `exchange_rate/exchange_rate.csv`            | 96        | 96 192 336 720    |
| `run_traffic.sh`       | `traffic/traffic.csv`                        | 96        | 96 192 336 720    |
| `run_weather.sh`       | `weather/weather.csv`                        | 96        | 96 192 336 720    |
| `run_illness.sh`       | `illness/national_illness.csv`               | 36        | 24 36 48 60       |
| `run_monash_to_etth1.sh` | Monash subset &rarr; `ETT-small/ETTh1.csv` | 96        | 96 192 336 720    |
| `run_monash_to_etth2.sh` | Monash subset &rarr; `ETT-small/ETTh2.csv` | 96        | 96 192 336 720    |

`run_electricity.sh` and `run_traffic.sh` also set `max_train=500000` to cap
training windows on the two very large multivariate datasets.

`run_monash_to_*.sh` pretrains on a subset of Monash `.tsf` files (per-window
RevIN z-score) then evaluates on a benchmark CSV's test split (Time-MoE
StandardScaler domain via de-RevIN, so MSE/MAE stays comparable).

## Customizing

Edit the variables at the top of any script:

```bash
train_root_path=/root/dataset/ETT-small/
train_data_path=ETTh2.csv          # train on ETTh2
test_root_path=/root/dataset/ETT-small/
test_data_path=ETTh1.csv           # evaluate on ETTh1
seq_len=512                        # longer context
```

and modify the `for pred_len in ...` loop to change the horizons.

For `run_monash_to_*.sh`, edit the `monash_files=(...)` array to pick which
Monash files to pretrain on; everything else works the same as the CSV
scripts.
