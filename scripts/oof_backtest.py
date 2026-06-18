"""Generate honest out-of-fold backtest metrics."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from eca.features.build import FEATURE_COLUMNS
from eca.backtest import run_backtest

df = pd.read_parquet("data/processed/features_labelled.parquet")
df["call_date"] = pd.to_datetime(df["call_date"])
df = df.sort_values("call_date").reset_index(drop=True)

cols = [c for c in FEATURE_COLUMNS if c in df.columns]
X = df[cols].astype(float).fillna(0.0)
y = df["label"].astype(int)

splitter = TimeSeriesSplit(n_splits=5)
oof = np.full(len(X), np.nan)

for tr, te in splitter.split(X):
    if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2:
        continue
    m = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_data_in_leaf=20, verbose=-1,
    )
    m.fit(X.iloc[tr], y.iloc[tr])
    oof[te] = m.predict_proba(X.iloc[te])[:, 1]

bt_df = df.assign(prob_up=oof).dropna(subset=["prob_up", "ret_excess"])
print(f"OOF rows: {len(bt_df)}")

for tau in [0.0, 0.10, 0.20]:
    r = run_backtest(bt_df, threshold=tau)
    s = r.summary()
    print(
        f"tau={tau}: trades={s['n_trades']} hit_rate={s['hit_rate']:.3f} "
        f"sharpe={s['annualised_sharpe']:.3f} maxdd={s['max_drawdown']:.3f} "
        f"bench_ret={s['benchmark_total_return']:.3f}"
    )
