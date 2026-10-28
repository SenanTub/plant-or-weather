"""
Plant or weather? How much of a plant's electrical signal is explained by
weather and the day-night cycle, and how much is left that is plant-specific.

Input: a tidy table of 10-minute values with columns
    time (datetime), plant (str), channel (str: 'stem'/'leaf'), signal (mV)
and a weather table of 10-minute values with columns
    time, air_temp (C), rh (%), irradiance (W/m2), wind (m/s), precip (mm)

Analyses
1. Daily cycle per plant (hour-of-day profile).
2. Out-of-sample variance explained (R^2) with day-blocked cross-validation:
     A  time of day only (Fourier terms)
     B  weather only (incl. VPD and 1-2 h lags)
     C  weather + time of day (linear)
     D  weather + time of day (gradient boosting, nonlinear ceiling)
   Done on the within-day signal (24 h rolling mean removed), so slow electrode
   drift doesn't masquerade as either weather or plant information.
3. Between-plant agreement: correlation of the within-day signal across plants,
   before and after removing the weather model's prediction.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STEP = "10min"


def vpd_kpa(temp_c, rh_pct):
    """Vapour pressure deficit (kPa), Tetens equation."""
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    return es * (1 - np.clip(rh_pct, 0, 100) / 100)


def weather_features(w):
    """Weather features on a regular 10-min index, including lags."""
    w = w.sort_index().copy()
    w["vpd"] = vpd_kpa(w["air_temp"], w["rh"])
    per_hour = pd.Timedelta("1h") // pd.Timedelta(STEP)
    for col in ["irradiance", "vpd", "air_temp"]:
        for lag_h in (1, 2):
            w[f"{col}_lag{lag_h}h"] = w[col].shift(lag_h * per_hour)
    # rain in the last 3 hours (wetting of plant surfaces / electrodes)
    w["precip_3h"] = w["precip"].rolling(3 * per_hour, min_periods=1).sum()
    return w


def time_features(index, harmonics=3):
    hour = index.hour + index.minute / 60
    out = {}
    for k in range(1, harmonics + 1):
        out[f"sin{k}"] = np.sin(2 * np.pi * k * hour / 24)
        out[f"cos{k}"] = np.cos(2 * np.pi * k * hour / 24)
    return pd.DataFrame(out, index=index)


def within_day(signal):
    """Remove a centred 24 h rolling mean: keeps daily dynamics, drops drift."""
    per_day = pd.Timedelta("1D") // pd.Timedelta(STEP)
    trend = signal.rolling(per_day, center=True, min_periods=per_day // 2).mean()
    return signal - trend


def cv_r2(X, y, groups, model_factory, n_splits=5):
    """Out-of-sample R^2 with whole days held out; returns R^2 and predictions."""
    pred = pd.Series(np.nan, index=y.index)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = model_factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        pred.iloc[te] = m.predict(X.iloc[te])
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot, pred


def linear():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))


def boosted():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=31, random_state=0)


WEATHER_COLS = ["air_temp", "rh", "vpd", "irradiance", "wind", "precip_3h",
                "irradiance_lag1h", "irradiance_lag2h", "vpd_lag1h", "vpd_lag2h",
                "air_temp_lag1h", "air_temp_lag2h"]


def variance_explained(sig, weather):
    """For each plant/channel: out-of-sample R^2 of models A-D on within-day signal."""
    wf = weather_features(weather)
    rows, residuals = [], {}
    for (plant, channel), s in sig.groupby(["plant", "channel"]):
        y = within_day(s.set_index("time")["signal"].asfreq(STEP))
        df = pd.concat([y.rename("y"), wf[WEATHER_COLS], time_features(y.index)],
                       axis=1, join="inner").dropna()
        if len(df) < 1000:
            continue
        groups = df.index.normalize()
        tcols = [c for c in df.columns if c.startswith(("sin", "cos"))]
        res = {"plant": plant, "channel": channel, "n_obs": len(df),
               "n_days": groups.nunique()}
        res["A_time_of_day"], _ = cv_r2(df[tcols], df["y"], groups, linear)
        res["B_weather"], pred_b = cv_r2(df[WEATHER_COLS], df["y"], groups, linear)
        res["C_weather_time"], _ = cv_r2(df[WEATHER_COLS + tcols], df["y"], groups, linear)
        res["D_boosted"], pred_d = cv_r2(df[WEATHER_COLS + tcols], df["y"], groups, boosted)
        rows.append(res)
        residuals[(plant, channel)] = pd.DataFrame({"y": df["y"], "resid": df["y"] - pred_d})
    return pd.DataFrame(rows), residuals


def between_plant_agreement(residuals, channel="stem"):
    """Mean pairwise correlation across plants, before vs after removing weather."""
    keys = [k for k in residuals if k[1] == channel]
    raw = pd.concat({k[0]: residuals[k]["y"] for k in keys}, axis=1).dropna()
    res = pd.concat({k[0]: residuals[k]["resid"] for k in keys}, axis=1).dropna()

    def mean_offdiag(c):
        m = c.values
        return m[~np.eye(len(m), dtype=bool)].mean()

    return {"raw_corr": raw.corr(), "resid_corr": res.corr(),
            "mean_raw_corr": mean_offdiag(raw.corr()),
            "mean_resid_corr": mean_offdiag(res.corr()),
            "n_obs_overlap": len(raw)}


def hourly_profile(sig, channel="stem"):
    """Median within-day signal by hour, per plant."""
    out = {}
    for (plant, ch), s in sig.groupby(["plant", "channel"]):
        if ch != channel:
            continue
        y = within_day(s.set_index("time")["signal"].asfreq(STEP)).dropna()
        out[plant] = y.groupby(y.index.hour).median()
    return pd.DataFrame(out)
