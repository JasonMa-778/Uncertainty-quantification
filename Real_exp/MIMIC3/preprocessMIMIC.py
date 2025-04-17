import pandas as pd
import numpy as np
import json
from collections import ChainMap
from os.path import join
import sys
from tqdm import tqdm
dayconverter = lambda x: np.timedelta64(x, 'ns') / np.timedelta64(1, 'D')
if __name__ == "__main__":
    path = sys.argv[1]
    print('[1/7] Reading chartevents file...')
    try:
        df = pd.read_csv(join(path, 'CHARTEVENTS.csv.gz'), compression='gzip',
                         usecols=['icustay_id', 'itemid', 'valuenum', 'charttime'])
    except FileNotFoundError:
        print(f"❌ 文件 {join(path, 'chartevents.csv.gz')} 不存在，请检查路径。")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 读取 chartevents.csv.gz 出错: {e}")
        sys.exit(1)
    df.dropna(inplace=True)
    print('[2/7] Computing admission and discharge times...')
    adm_dates = df.groupby('icustay_id')['charttime'].min()
    exit_dates = df.groupby('icustay_id')['charttime'].max()
    print('[3/7] Calculating ICU stay duration...')
    icudates = pd.DataFrame({'intime': adm_dates, 'outtime': exit_dates})
    icudates['duration'] = pd.to_datetime(icudates['outtime']) - pd.to_datetime(icudates['intime'])
    icudates = icudates[
        (icudates['duration'] >= pd.Timedelta(days=1)) &
        (icudates['duration'] <= pd.Timedelta(days=2))
    ]
    print('[4/7] Filtering ICU stays not in [1, 2] days...')
    df = df[df['icustay_id'].isin(icudates.index)]
    df['dt'] = pd.to_datetime(df['charttime']) - pd.to_datetime(df['icustay_id'].map(icudates['intime']))
    print('[5/7] Filtering events outside the first 3 hours...')
    df = df[df['dt'] < pd.Timedelta(hours=3)]
    df['dt'] = df['dt'].dt.total_seconds() / 60
    pdf = df.groupby(["icustay_id", "dt"]).apply(lambda x: list(zip(x.itemid, x.valuenum))).reset_index(name='val')
    pdf = pdf.groupby("icustay_id").apply(lambda x: dict(zip(x.dt, x.val))).to_dict()
    print('[6/7] Keeping all records without event limits...')
    res = {}
    for patient, data in tqdm(pdf.items(), desc="Processing events"):
        res[int(patient)] = data
    print('[7/7] Reading admissions and icustays...')
    try:
        adm = pd.read_csv(join(path, 'ADMISSIONS.csv.gz'), compression='gzip', usecols=['hadm_id', 'deathtime'])
        icustays = pd.read_csv(join(path, 'ICUSTAYS.csv.gz'), compression='gzip', usecols=['hadm_id', 'icustay_id'])
    except Exception as e:
        print(f"❌ 读取 admissions/icustays 文件失败: {e}")
        sys.exit(1)
    icustay2hadm = dict(zip(icustays.icustay_id, icustays.hadm_id))
    adm['deathtime'] = adm.deathtime.isna().astype(int)
    hadm2label = dict(zip(adm.hadm_id, adm.deathtime))
    icustay2label = {icu_k: hadm2label.get(hadm_k, 0) for icu_k, hadm_k in icustay2hadm.items()}
    restricted_icustay2label = {k: v for k, v in icustay2label.items() if k in res}
    print('Saving JSON files...')
    with open('events.json', "w") as f:
        json.dump(res, f, sort_keys=True)
    with open('targets.json', "w") as f:
        json.dump(restricted_icustay2label, f, sort_keys=True)
    print("✅ Preprocessing complete!")