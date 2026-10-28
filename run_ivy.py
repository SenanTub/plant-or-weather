import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import analysis as A

P = pd.read_csv("/home/claude/work/plants_10min.csv", parse_dates=["datetime"])
W = pd.read_csv("/home/claude/work/weather_10min.csv", parse_dates=["time"])
W["time"] = W["time"] - pd.Timedelta("10min")   # station stamps the END of each 10-min interval
W = W.set_index("time").asfreq("10min")
# keep only plant-days with >=80% coverage (same rule as the paper)
P["day"] = P.datetime.dt.normalize()
cov = P.groupby(["plant", "day"]).size() / 144
good = cov[cov >= 0.8].index
P = P.set_index(["plant", "day"]).loc[good].reset_index()
sig = pd.concat([
    P.rename(columns={"datetime": "time", "CH1_mean": "signal"}).assign(channel="CH1")[["time", "plant", "channel", "signal"]],
    P.rename(columns={"datetime": "time", "CH2_mean": "signal"}).assign(channel="CH2")[["time", "plant", "channel", "signal"]],
])
sig["signal"] = sig["signal"] / 1e3   # thousands of ADC counts, for readable numbers
ve, resid = A.variance_explained(sig, W)
pd.set_option("display.width", 200)
print(ve.round(3).to_string(index=False))
ve.to_csv("/home/claude/work/variance_explained.csv", index=False)
for ch in ["CH1", "CH2"]:
    ag = A.between_plant_agreement(resid, ch)
    print(ch, "overlap obs", ag["n_obs_overlap"], "mean corr raw", round(ag["mean_raw_corr"], 3),
          "after weather", round(ag["mean_resid_corr"], 3))
    print(ag["raw_corr"].round(2))
pd.to_pickle(resid, "/home/claude/work/resid.pkl")
