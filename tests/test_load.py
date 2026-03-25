import sys
import os
from datasets import load_dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.data_loader import load_cord_sample
import json

def extraer_total(sample):
    gt = json.loads(sample["ground_truth"])
    return gt["gt_parse"]["total"]["total_price"]


dataset = load_dataset("naver-clova-ix/cord-v2", split="train", streaming=True)

for i, sample in enumerate(dataset):
    total = extraer_total(sample)
    print("TOTAL FACTURA:", total)
    if i == 0: break
    print("\n"*3)

