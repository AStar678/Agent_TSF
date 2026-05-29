"""Global paths and default hyper-parameters."""
import os

# ---- Data paths ----
MONASH_DIR = "/root/monash_tsf_all/data"
GIFT_DIR = "/root/gift-eval/Mini-GiftPretrain"
ETTH1_CSV = "/root/ETTh1.csv"
ETTH2_CSV = "/root/ETTh2.csv"
WEATHER_CSV = "/root/weather.csv"

# ---- Weights ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ---- Default hyper-parameters ----
DEFAULT_KERNELS = [5, 13, 25, 49, 97]
DEFAULT_L_IN = 96
DEFAULT_L_OUT = 96
DEFAULT_RIDGE = 1e-4
DEFAULT_MAX_PER_SERIES = 200
DEFAULT_OUTLIER_THR = 10.0
DEFAULT_MIN_STD = 1e-6
