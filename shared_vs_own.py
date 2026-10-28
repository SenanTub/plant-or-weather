import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
X = pd.read_csv("/home/claude/work/hourly_features.csv", parse_dates=["hour_start"])
W = pd.read_csv("/home/claude/work/weather_10min.csv", parse_dates=["time"])
W["time"] -= pd.Timedelta("10min"); W = W.set_index("time")
import sys; sys.path.insert(0, "/home/claude/plant_or_weather"); from analysis import vpd_kpa
W["vpd"] = vpd_kpa(W.air_temp, W.rh)
H = W[["irradiance","air_temp","rh","vpd","wind","precip"]].resample("1h").agg(
    {"irradiance":"mean","air_temp":"mean","rh":"mean","vpd":"mean","wind":"mean","precip":"sum"})
out = []
for ch in ["CH1","CH2"]:
    for feat in ["logstd", "level"]:
        T = X.pivot(index="hour_start", columns="plant", values=f"{ch}_{feat}")
        if feat == "level":   # remove each plant's daily mean so slow drift doesn't count
            T = T - T.groupby(T.index.normalize()).transform("mean")
        else:
            T = T - T.rolling("7D", center=True).median()
        for p in T.columns:
            others = T.drop(columns=p)
            d = pd.concat([T[p].rename("y"), others.mean(axis=1).rename("others")], axis=1).dropna()
            d = d[others.loc[d.index].notna().sum(axis=1) >= 2]
            r = np.corrcoef(d.y, d.others)[0, 1]
            out.append((ch, feat, p, len(d), round(r, 2), round(r**2, 2)))
R = pd.DataFrame(out, columns=["channel","feature","plant","hours","r_with_other_plants","R2"])
print(R.to_string(index=False))
print(R.groupby(["channel","feature"])[["r_with_other_plants","R2"]].mean().round(2))
R.to_csv("/home/claude/work/shared_vs_own.csv", index=False)

# drivers of variability: standardised correlation of within-week log-std with each weather variable
T = X.pivot(index="hour_start", columns="plant", values="CH1_logstd")
T = T - T.rolling("7D", center=True).median()
D = pd.concat([T.mean(axis=1).rename("mean_logstd"), H], axis=1).dropna()
print("hours:", len(D)); print(D.corr(method="spearman")["mean_logstd"].round(2))
