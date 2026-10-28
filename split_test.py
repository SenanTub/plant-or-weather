"""Day/night classification from 1-hour windows: does the split method change the score?"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import f1_score

P = pd.read_csv("/home/claude/work/plants_10min.csv", parse_dates=["datetime"])
W = pd.read_csv("/home/claude/work/weather_10min.csv", parse_dates=["time"])
W["time"] -= pd.Timedelta("10min"); W = W.set_index("time").asfreq("10min")
P["day"] = P.datetime.dt.normalize()
cov = P.groupby(["plant","day"]).size()/144; good = cov[cov>=0.8].index
P = P.set_index(["plant","day"]).loc[good].reset_index()
P["hour_start"] = P.datetime.dt.floor("1h")

def feats(g):
    out = {}
    for ch in ["CH1","CH2"]:
        m = g[f"{ch}_mean"].values; s = np.log(g[f"{ch}_std"].clip(lower=1).values)
        out[f"{ch}_level"] = m.mean()
        out[f"{ch}_range"] = m.max() - m.min()
        out[f"{ch}_slope"] = np.polyfit(np.arange(len(m)), m, 1)[0] if len(m) > 1 else 0
        out[f"{ch}_logstd"] = s.mean()
        out[f"{ch}_logstd_sd"] = s.std()
    out["n"] = len(g)
    return pd.Series(out)

X = P.groupby(["plant","hour_start"]).apply(feats)
X = X[X.n >= 5].drop(columns="n").reset_index()
irr = W.irradiance.resample("1h").mean()
X["irr"] = irr.reindex(X.hour_start).values
X = X.dropna()
y = (X.irr > 50).astype(int).values
day = X.hour_start.dt.normalize().values
plant = X.plant.values

def run(cols, splitter, groups=None, name=""):
    scores = []
    for tr, te in splitter.split(X[cols], y, groups):
        m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0, class_weight="balanced")
        m.fit(X[cols].iloc[tr], y[tr])
        scores.append(f1_score(y[te], m.predict(X[cols].iloc[te]), average="macro"))
    return np.mean(scores), np.std(scores)

all_cols = [c for c in X.columns if c.startswith(("CH1","CH2"))]
level_cols = [c for c in all_cols if c.endswith("level")]
var_cols = [c for c in all_cols if "logstd" in c]
res = []
for label, cols in [("all features", all_cols), ("level only", level_cols), ("variability only", var_cols)]:
    for sname, sp, grp in [
        ("shuffled 1-h windows (as in paper)", StratifiedShuffleSplit(10, test_size=0.2, random_state=0), None),
        ("whole days held out", GroupKFold(5), day),
        ("whole plant held out", LeaveOneGroupOut(), plant)]:
        mu, sd = run(cols, sp, grp)
        res.append((label, sname, round(mu, 3), round(sd, 3)))
R = pd.DataFrame(res, columns=["features", "split", "macro_F1", "sd"])
print("windows:", len(X), "day share:", round(y.mean(), 3))
print(R.to_string(index=False))
R.to_csv("/home/claude/work/split_test.csv", index=False)
X.to_csv("/home/claude/work/hourly_features.csv", index=False)
