"""Step 2 - Feature engineering: features capture CHANGE, not just level (paper Sec. IV-C)."""
import pandas as pd

SENSOR = ["ndvi", "temp_gap", "red_edge"]
WEATHER = ["temp", "humidity", "leaf_wet_h"]


def build_features(df):
    df = df.sort_values(["field", "zone", "day"]).copy()
    g = df.groupby(["field", "zone"])
    for c in SENSOR:
        df[c + "_d1"] = g[c].diff(1)                       # day-to-day change
        for k in (3, 5, 7):
            df[f"{c}_slope{k}"] = (df[c] - g[c].shift(k)) / k   # trend over k days
    df["wet_3d"] = g["leaf_wet_h"].transform(lambda s: s.rolling(3).sum())
    df["hum_mean_5d"] = g["humidity"].transform(lambda s: s.rolling(5).mean())
    df["fav_days_5d"] = g["leaf_wet_h"].transform(lambda s: (s >= 7).astype(float).rolling(5).sum())
    df = df.dropna().reset_index(drop=True)
    sensor_f = [c for c in df.columns if c.split("_slope")[0].split("_d1")[0] in SENSOR]
    weather_f = WEATHER + ["wet_3d", "hum_mean_5d", "fav_days_5d"]
    meta_f = ["susceptibility"]
    return df, sensor_f, weather_f, meta_f
