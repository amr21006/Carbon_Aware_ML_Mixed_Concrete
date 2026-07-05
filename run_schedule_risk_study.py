"""Open-data schedule risk study for NYC capital projects.

This script builds a reviewer-oriented empirical pipeline for predicting
near-term forecast completion slippage in public capital projects.

Main target:
    At reporting period t, predict whether the same project will record a
    forecast completion extension greater than a selected threshold at the next
    reporting period.

Data sources:
    - NYC Capital Projects Dashboard: Citywide Budget and Schedule (fb86-vt7u)
    - NYC Capital Projects Dashboard: Citywide Schedule History (95tx-snak)
    - NOAA ISD Global Hourly, LaGuardia station 72503014732
    - NOAA Storm Events bulk CSV files
"""

from __future__ import annotations

import gzip
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

NYC_SCHEDULE_URL = (
    "https://data.cityofnewyork.us/resource/95tx-snak.csv?$limit=500000"
)
NYC_BUDGET_URL = (
    "https://data.cityofnewyork.us/resource/fb86-vt7u.csv?$limit=500000"
)
NOAA_ISD_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/72503014732.csv"
NOAA_STORM_DIR = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

NYC_COUNTY_KEYWORDS = ("BRONX", "KINGS", "NEW YORK", "QUEENS", "RICHMOND")
MODELING_YEARS = (2023, 2024, 2025, 2026)


@dataclass(frozen=True)
class ExperimentConfig:
    threshold_days: int
    target_name: str


def ensure_dirs() -> None:
    for path in (RAW, PROCESSED, RESULTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def download_csv(url: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        df = pd.read_csv(url)
        df.to_csv(path, index=False)
    return pd.read_csv(path, low_memory=False)


def parse_yyyymm(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    text = str(int(float(value))) if str(value).replace(".", "", 1).isdigit() else str(value)
    return pd.to_datetime(text + "01", format="%Y%m%d", errors="coerce")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_isd_temp(value: object) -> float:
    if pd.isna(value):
        return np.nan
    token = str(value).split(",")[0]
    if token in ("+9999", "9999", "-9999", ""):
        return np.nan
    return pd.to_numeric(token, errors="coerce") / 10.0


def parse_isd_wind_speed(value: object) -> float:
    if pd.isna(value):
        return np.nan
    parts = str(value).split(",")
    if len(parts) < 4:
        return np.nan
    token = parts[3]
    if token in ("9999", "+9999", ""):
        return np.nan
    return pd.to_numeric(token, errors="coerce") / 10.0


def parse_isd_precip(value: object) -> float:
    if pd.isna(value):
        return np.nan
    parts = str(value).split(",")
    if len(parts) < 2:
        return np.nan
    depth = parts[1]
    if depth in ("9999", "+9999", ""):
        return np.nan
    return max(float(pd.to_numeric(depth, errors="coerce")) / 10.0, 0.0)


def build_weather_daily() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in MODELING_YEARS:
        path = RAW / f"noaa_isd_lga_{year}.csv"
        if not path.exists():
            try:
                df_year = pd.read_csv(NOAA_ISD_URL.format(year=year), low_memory=False)
            except (HTTPError, URLError):
                continue
            df_year.to_csv(path, index=False)
        df = pd.read_csv(path, low_memory=False)
        keep = [c for c in ["DATE", "TMP", "DEW", "WND", "AA1", "AA2", "AA3", "AA4"] if c in df.columns]
        df = df[keep].copy()
        df["datetime"] = pd.to_datetime(df["DATE"], errors="coerce")
        df["date"] = df["datetime"].dt.floor("D")
        df["temp_c"] = df["TMP"].map(parse_isd_temp)
        df["dew_c"] = df["DEW"].map(parse_isd_temp) if "DEW" in df else np.nan
        df["wind_mps"] = df["WND"].map(parse_isd_wind_speed) if "WND" in df else np.nan
        precip_cols = [c for c in ["AA1", "AA2", "AA3", "AA4"] if c in df.columns]
        if precip_cols:
            precip = pd.concat([df[c].map(parse_isd_precip) for c in precip_cols], axis=1)
            df["precip_mm"] = precip.max(axis=1).fillna(0.0)
        else:
            df["precip_mm"] = 0.0
        frames.append(df)

    hourly = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    daily = hourly.groupby("date").agg(
        temp_c_mean=("temp_c", "mean"),
        temp_c_max=("temp_c", "max"),
        temp_c_min=("temp_c", "min"),
        dew_c_mean=("dew_c", "mean"),
        wind_mps_mean=("wind_mps", "mean"),
        wind_mps_max=("wind_mps", "max"),
        precip_mm_sum=("precip_mm", "sum"),
        hot_hour_count=("temp_c", lambda s: int((s >= 32.0).sum())),
        freezing_hour_count=("temp_c", lambda s: int((s <= 0.0).sum())),
        high_wind_hour_count=("wind_mps", lambda s: int((s >= 10.0).sum())),
    ).reset_index()
    daily.to_csv(PROCESSED / "weather_daily_lga.csv", index=False)
    return daily


def latest_storm_file_for_year(index_html: str, year: int) -> str:
    pattern = rf'StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz'
    matches = re.findall(pattern, index_html)
    if not matches:
        raise RuntimeError(f"No NOAA Storm Events file found for {year}")
    return sorted(set(matches))[-1]


def parse_damage_amount(value: object) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip().upper()
    if not text or text in {"0", "0.00", "K", "M", "B"}:
        return 0.0
    multiplier = 1.0
    if text.endswith("K"):
        multiplier = 1_000.0
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text.endswith("B"):
        multiplier = 1_000_000_000.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return 0.0


def build_storm_daily() -> pd.DataFrame:
    index_path = RAW / "noaa_stormevents_index.html"
    if not index_path.exists():
        index_path.write_text(urlopen(NOAA_STORM_DIR, timeout=60).read().decode("utf-8"), encoding="utf-8")
    index_html = index_path.read_text(encoding="utf-8")

    frames: list[pd.DataFrame] = []
    for year in MODELING_YEARS:
        filename = latest_storm_file_for_year(index_html, year)
        gz_path = RAW / filename
        if not gz_path.exists():
            gz_path.write_bytes(urlopen(NOAA_STORM_DIR + filename, timeout=120).read())
        with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as fh:
            df = pd.read_csv(fh, low_memory=False)
        df.columns = [c.upper() for c in df.columns]
        if "STATE" not in df or "CZ_NAME" not in df:
            continue
        df = df[df["STATE"].astype(str).str.upper().eq("NEW YORK")].copy()
        county_text = df["CZ_NAME"].astype(str).str.upper()
        county_mask = np.logical_or.reduce([county_text.str.contains(k, regex=False) for k in NYC_COUNTY_KEYWORDS])
        df = df[county_mask].copy()
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df.get("BEGIN_DATE_TIME"), errors="coerce").dt.floor("D")
        if df["date"].isna().all() and "BEGIN_YEARMONTH" in df and "BEGIN_DAY" in df:
            df["date"] = pd.to_datetime(
                df["BEGIN_YEARMONTH"].astype(str) + df["BEGIN_DAY"].astype(str).str.zfill(2),
                format="%Y%m%d",
                errors="coerce",
            )
        df["damage_property_usd"] = df.get("DAMAGE_PROPERTY", pd.Series(index=df.index)).map(parse_damage_amount)
        event = df.get("EVENT_TYPE", pd.Series("", index=df.index)).astype(str).str.upper()
        df["flood_event"] = event.str.contains("FLOOD|FLASH FLOOD|COASTAL FLOOD", regex=True).astype(int)
        df["wind_event"] = event.str.contains("WIND|TORNADO|THUNDERSTORM", regex=True).astype(int)
        df["winter_event"] = event.str.contains("SNOW|ICE|WINTER|BLIZZARD", regex=True).astype(int)
        df["heat_event"] = event.str.contains("HEAT", regex=False).astype(int)
        frames.append(df)

    if not frames:
        daily = pd.DataFrame({"date": pd.to_datetime([])})
    else:
        events = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
        daily = events.groupby("date").agg(
            storm_event_count=("EVENT_TYPE", "size"),
            storm_type_count=("EVENT_TYPE", "nunique"),
            flood_event_count=("flood_event", "sum"),
            wind_event_count=("wind_event", "sum"),
            winter_event_count=("winter_event", "sum"),
            heat_event_count=("heat_event", "sum"),
            storm_damage_property_usd=("damage_property_usd", "sum"),
        ).reset_index()
    daily.to_csv(PROCESSED / "storm_daily_nyc.csv", index=False)
    return daily


def rolling_exposure(
    observation_dates: Iterable[pd.Timestamp],
    daily: pd.DataFrame,
    date_col: str,
    feature_cols: list[str],
    windows: tuple[int, ...],
    prefix: str,
) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(list(observation_dates))).drop_duplicates().sort_values()
    out = pd.DataFrame({"data_date": dates})
    if daily.empty:
        for window in windows:
            for col in feature_cols:
                out[f"{prefix}_{col}_{window}d"] = 0.0
        return out

    daily = daily.copy()
    daily[date_col] = pd.to_datetime(daily[date_col])
    daily = daily.set_index(date_col).sort_index()
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx)
    for col in feature_cols:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)

    for window in windows:
        rolled = daily[feature_cols].rolling(f"{window}D", min_periods=1).agg(["sum", "mean", "max"])
        rolled.columns = [f"{prefix}_{col}_{stat}_{window}d" for col, stat in rolled.columns]
        rolled = rolled.reset_index().rename(columns={"index": "data_date"})
        out = out.merge(rolled, on="data_date", how="left")
    return out.fillna(0.0)


def mode_or_first(series: pd.Series) -> object:
    series = series.dropna()
    if series.empty:
        return np.nan
    modes = series.mode()
    return modes.iloc[0] if not modes.empty else series.iloc[0]


def prepare_project_data() -> pd.DataFrame:
    schedule = download_csv(NYC_SCHEDULE_URL, RAW / "nyc_schedule_history_95tx_snak.csv")
    budget = download_csv(NYC_BUDGET_URL, RAW / "nyc_budget_schedule_fb86_vt7u.csv")

    for df in (schedule, budget):
        df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct") for c in df.columns]
        df["reporting_period"] = numeric(df["reporting_period"])
        df["pid"] = numeric(df["pid"])
        df["managing_agency"] = df["managing_agency"].astype(str).str.strip()

    budget_numeric = ["total_budget", "spend_to_date", "spend_to_date_pct"]
    for col in budget_numeric:
        if col in budget:
            budget[col] = numeric(budget[col])
    date_cols = [
        "current_phase_start",
        "forecast_current_phase_end",
        "forecast_completion",
        "actual_design_start",
        "actual_design_end",
        "actual_construction_procurement_start",
        "actual_construction_procurement_end",
        "actual_construction_start",
        "actual_construction_end",
    ]
    for col in date_cols:
        if col in budget:
            budget[col] = pd.to_datetime(budget[col], errors="coerce")

    agg_spec = {
        "total_budget": "sum",
        "spend_to_date": "sum",
        "spend_to_date_pct": "mean",
        "fms_id": "nunique",
        "sponsor_agency": mode_or_first,
        "borough": mode_or_first,
        "community_board": mode_or_first,
        "ten_year_plan_category": mode_or_first,
        "budget_line": "nunique",
        "forecast_completion": "max",
        "current_phase_start": "min",
        "forecast_current_phase_end": "max",
        "actual_design_start": "min",
        "actual_design_end": "max",
        "actual_construction_start": "min",
        "actual_construction_end": "max",
    }
    agg_spec = {k: v for k, v in agg_spec.items() if k in budget.columns}
    budget_agg = (
        budget.groupby(["reporting_period", "managing_agency", "pid"], dropna=False)
        .agg(agg_spec)
        .reset_index()
        .rename(columns={"fms_id": "fms_line_count", "budget_line": "budget_line_count"})
    )

    schedule["completion_date"] = pd.to_datetime(schedule["completion_date"], errors="coerce")
    schedule["data_date"] = pd.to_datetime(schedule["data_date"], errors="coerce")
    schedule["variance_day"] = numeric(schedule["variance_day"])
    schedule["report_month_date"] = schedule["reporting_period"].map(parse_yyyymm)
    schedule = schedule.sort_values(["managing_agency", "pid", "reporting_period", "data_date"])
    group_cols = ["managing_agency", "pid"]
    schedule["project_key"] = schedule["managing_agency"].astype(str) + "_" + schedule["pid"].astype("Int64").astype(str)
    schedule["next_variance_day"] = schedule.groupby(group_cols)["variance_day"].shift(-1)
    schedule["next_reporting_period"] = schedule.groupby(group_cols)["reporting_period"].shift(-1)
    schedule["next_data_date"] = schedule.groupby(group_cols)["data_date"].shift(-1)
    schedule["lag1_variance_day"] = schedule.groupby(group_cols)["variance_day"].shift(1)
    schedule["lag2_variance_day"] = schedule.groupby(group_cols)["variance_day"].shift(2)
    schedule["rolling_mean_variance_day"] = (
        schedule.groupby(group_cols)["variance_day"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    schedule["prior_slip_count"] = (
        schedule.groupby(group_cols)["variance_day"]
        .transform(lambda s: (s.shift(1) > 30).expanding(min_periods=1).sum())
    )
    schedule["project_obs_index"] = schedule.groupby(group_cols).cumcount()

    df = schedule.merge(
        budget_agg,
        on=["reporting_period", "managing_agency", "pid"],
        how="left",
        suffixes=("", "_budget"),
    )

    weather_daily = build_weather_daily()
    storm_daily = build_storm_daily()
    obs_dates = df["data_date"].dropna().dt.floor("D").unique()
    weather_features = rolling_exposure(
        obs_dates,
        weather_daily,
        "date",
        [
            "temp_c_mean",
            "temp_c_max",
            "temp_c_min",
            "wind_mps_max",
            "precip_mm_sum",
            "hot_hour_count",
            "freezing_hour_count",
            "high_wind_hour_count",
        ],
        (30, 90),
        "weather",
    )
    storm_features = rolling_exposure(
        obs_dates,
        storm_daily,
        "date",
        [
            "storm_event_count",
            "storm_type_count",
            "flood_event_count",
            "wind_event_count",
            "winter_event_count",
            "heat_event_count",
            "storm_damage_property_usd",
        ],
        (30, 90),
        "storm",
    )

    df["data_date"] = df["data_date"].dt.floor("D")
    df = df.merge(weather_features, on="data_date", how="left")
    df = df.merge(storm_features, on="data_date", how="left")

    df["report_year"] = df["report_month_date"].dt.year
    df["report_month"] = df["report_month_date"].dt.month
    df["report_quarter"] = df["report_month_date"].dt.quarter
    df["days_to_completion"] = (df["completion_date"] - df["data_date"]).dt.days
    if "current_phase_start" in df:
        df["days_in_current_phase"] = (df["data_date"] - df["current_phase_start"]).dt.days
    else:
        df["days_in_current_phase"] = np.nan
    if "forecast_current_phase_end" in df:
        df["days_to_phase_end"] = (df["forecast_current_phase_end"] - df["data_date"]).dt.days
    else:
        df["days_to_phase_end"] = np.nan
    df["total_budget_log"] = np.log1p(df.get("total_budget", 0).fillna(0))
    df["spend_to_date_log"] = np.log1p(df.get("spend_to_date", 0).fillna(0))
    df["spend_ratio"] = df.get("spend_to_date", 0) / df.get("total_budget", np.nan).replace({0: np.nan})
    df["variance_day_clipped"] = df["variance_day"].clip(lower=-365, upper=365)
    df["lag1_variance_day_clipped"] = df["lag1_variance_day"].clip(lower=-365, upper=365)
    df["lag2_variance_day_clipped"] = df["lag2_variance_day"].clip(lower=-365, upper=365)
    df["rolling_mean_variance_day_clipped"] = df["rolling_mean_variance_day"].clip(lower=-365, upper=365)
    df["current_slip_positive"] = (df["variance_day"] > 0).astype(int)
    reason = df.get("reason_for_forecast_completion_change", pd.Series("", index=df.index)).fillna("").astype(str)
    reason_upper = reason.str.upper()
    project_name = df.get("agency_project_name", pd.Series("", index=df.index)).fillna("").astype(str)
    df["reason_length"] = reason.str.len()
    df["reason_missing"] = (reason.str.len() == 0).astype(int)
    df["reason_weather_keyword"] = reason_upper.str.contains("WEATHER|STORM|RAIN|FLOOD|SNOW|HURRICANE", regex=True).astype(int)
    df["reason_procurement_keyword"] = reason_upper.str.contains("PROCUREMENT|BID|CONTRACT|VENDOR", regex=True).astype(int)
    df["reason_scope_keyword"] = reason_upper.str.contains("SCOPE|CHANGE|DESIGN|FIELD|SITE", regex=True).astype(int)
    df["text_feature"] = (project_name + " " + reason).str.strip().replace("", "missing_text")

    for col in [
        "actual_design_start",
        "actual_design_end",
        "actual_construction_start",
        "actual_construction_end",
    ]:
        df[f"has_{col}"] = int(0)
        if col in df:
            df[f"has_{col}"] = df[col].notna().astype(int)

    df.to_csv(PROCESSED / "modeling_base.csv", index=False)
    return df


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = {
        "target",
        "next_variance_day",
        "next_reporting_period",
        "next_data_date",
        "agency_project_name",
        "reason_for_forecast_completion_change",
        "completion_date",
        "data_date",
        "report_month_date",
        "project_key",
        "pid",
        "text_feature",
    }
    categorical = [
        c
        for c in [
            "managing_agency",
            "current_phase",
            "completion_date_type",
            "sponsor_agency",
            "borough",
            "community_board",
            "ten_year_plan_category",
            "report_month",
            "report_quarter",
        ]
        if c in df.columns
    ]
    numeric_cols = []
    for col in df.columns:
        if col in excluded or col in categorical:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    text_cols = ["text_feature"] if "text_feature" in df.columns else []
    return numeric_cols, categorical, text_cols


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str], text_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    transformers = [
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ]
    if text_cols:
        transformers.append(
            (
                "text",
                TfidfVectorizer(
                    max_features=800,
                    min_df=3,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
                text_cols[0],
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def build_models(pos_rate: float) -> dict[str, object]:
    neg_pos_ratio = (1.0 - pos_rate) / max(pos_rate, 1e-6)
    models: dict[str, object] = {
        "logistic_l2_balanced": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=42,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=450,
            max_depth=14,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=450,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            scale_pos_weight=neg_pos_ratio,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=31,
            min_child_samples=25,
            subsample=0.9,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
    if CatBoostClassifier is not None:
        models["catboost"] = CatBoostClassifier(
            iterations=450,
            depth=5,
            learning_rate=0.04,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=42,
        )
    return models


def safe_metric(fn, *args, default=np.nan, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def threshold_from_train(y_train: pd.Series, p_train: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(y_train, p_train >= t) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def evaluate_predictions(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": safe_metric(roc_auc_score, y_true, proba),
        "average_precision": safe_metric(average_precision_score, y_true, proba),
        "brier": safe_metric(brier_score_loss, y_true, proba),
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def bootstrap_ci(y_true: pd.Series, proba: np.ndarray, threshold: float, n_boot: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    y = np.asarray(y_true).astype(int)
    rows = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        rows.append(evaluate_predictions(pd.Series(y[idx]), proba[idx], threshold))
    boot = pd.DataFrame(rows)
    out = []
    for metric in ["accuracy", "balanced_accuracy", "roc_auc", "average_precision", "f1", "recall", "specificity"]:
        out.append(
            {
                "metric": metric,
                "ci_low": boot[metric].quantile(0.025),
                "ci_high": boot[metric].quantile(0.975),
            }
        )
    return pd.DataFrame(out)


def run_experiment(df: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["target"] = (data["next_variance_day"] > config.threshold_days).astype(int)
    data = data[data["next_variance_day"].notna()].copy()
    data = data[data["next_reporting_period"].notna()].copy()
    data = data.dropna(subset=["reporting_period", "project_key"])

    numeric_cols, categorical_cols, text_cols = feature_columns(data)
    X = data[numeric_cols + categorical_cols + text_cols]
    y = data["target"].astype(int)
    groups = data["project_key"]
    pos_rate = float(y.mean())

    preprocessor = make_preprocessor(numeric_cols, categorical_cols, text_cols)
    models = build_models(pos_rate)
    results = []
    predictions = []

    temporal_train = data["reporting_period"] <= 202412
    temporal_test = data["reporting_period"] >= 202501
    if temporal_test.sum() == 0 or y[temporal_test].nunique() < 2:
        temporal_train = data["reporting_period"] <= data["reporting_period"].quantile(0.70)
        temporal_test = ~temporal_train

    split_defs = [("temporal_2025_2026_holdout", temporal_train, temporal_test)]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    group_train = pd.Series(False, index=data.index)
    group_test = pd.Series(False, index=data.index)
    group_train.iloc[train_idx] = True
    group_test.iloc[test_idx] = True
    split_defs.append(("project_group_holdout", group_train, group_test))

    for split_name, train_mask, test_mask in split_defs:
        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        for model_name, model in models.items():
            pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            if hasattr(pipe.named_steps["model"], "predict_proba"):
                p_train = pipe.predict_proba(X_train)[:, 1]
                p_test = pipe.predict_proba(X_test)[:, 1]
            else:
                p_train = pipe.decision_function(X_train)
                p_test = pipe.decision_function(X_test)
            threshold = threshold_from_train(y_train, p_train)
            row = evaluate_predictions(y_test, p_test, threshold)
            row.update(
                {
                    "target": config.target_name,
                    "threshold_days": config.threshold_days,
                    "split": split_name,
                    "model": model_name,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "train_positive_rate": float(y_train.mean()),
                    "test_positive_rate": float(y_test.mean()),
                    "n_features_numeric": len(numeric_cols),
                    "n_features_categorical": len(categorical_cols),
                    "n_features_text": len(text_cols),
                }
            )
            results.append(row)
            predictions.append(
                pd.DataFrame(
                    {
                        "target": config.target_name,
                        "split": split_name,
                        "model": model_name,
                        "project_key": data.loc[test_mask, "project_key"].values,
                        "reporting_period": data.loc[test_mask, "reporting_period"].values,
                        "y_true": y_test.values,
                        "proba": p_test,
                        "threshold": threshold,
                    }
                )
            )

    results_df = pd.DataFrame(results).sort_values(
        ["split", "roc_auc", "balanced_accuracy"], ascending=[True, False, False]
    )
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()

    best_temporal = results_df[results_df["split"].eq("temporal_2025_2026_holdout")].head(1)
    ci_df = pd.DataFrame()
    if not best_temporal.empty and not pred_df.empty:
        best = best_temporal.iloc[0]
        best_pred = pred_df[
            pred_df["target"].eq(best["target"])
            & pred_df["split"].eq(best["split"])
            & pred_df["model"].eq(best["model"])
        ]
        ci_df = bootstrap_ci(best_pred["y_true"], best_pred["proba"].to_numpy(), float(best["threshold"]))
        ci_df["target"] = config.target_name
        ci_df["split"] = best["split"]
        ci_df["model"] = best["model"]

    return results_df, pred_df, ci_df


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = df[df["next_variance_day"].notna()].copy()
    rows.append({"item": "modeled_rows_with_next_variance", "value": len(base)})
    rows.append({"item": "unique_projects", "value": base["project_key"].nunique()})
    rows.append({"item": "unique_agencies", "value": base["managing_agency"].nunique()})
    rows.append({"item": "reporting_period_min", "value": int(base["reporting_period"].min())})
    rows.append({"item": "reporting_period_max", "value": int(base["reporting_period"].max())})
    for threshold in [0, 30, 60, 90]:
        rows.append(
            {
                "item": f"next_variance_gt_{threshold}_rate",
                "value": float((base["next_variance_day"] > threshold).mean()),
            }
        )
    rows.append({"item": "budget_rows_joined_rate", "value": float(base["total_budget"].notna().mean())})
    weather_cols = [c for c in base.columns if c.startswith("weather_")]
    storm_cols = [c for c in base.columns if c.startswith("storm_")]
    rows.append({"item": "weather_feature_count", "value": len(weather_cols)})
    rows.append({"item": "storm_feature_count", "value": len(storm_cols)})
    return pd.DataFrame(rows)


def write_model_card(results: pd.DataFrame, ci: pd.DataFrame, dataset_summary: pd.DataFrame) -> None:
    best = results[results["split"].eq("temporal_2025_2026_holdout")].sort_values(
        ["roc_auc", "balanced_accuracy"], ascending=False
    ).head(1)
    lines = [
        "# Schedule Risk Model Card",
        "",
        "Study: Open-data early warning model for schedule slippage in NYC public capital projects.",
        "",
        "Main validation design: temporal holdout, training on reporting periods up to 2024-12 and testing on 2025-2026 reporting periods.",
        "",
        "A secondary project-group holdout is also reported to test project-level leakage risk.",
        "",
        "The reported target is forward-looking: using reporting-period t features to predict whether the next observed reporting period records a forecast completion extension above the configured day threshold.",
        "",
        "## Dataset Summary",
        "",
        dataset_summary.to_markdown(index=False),
        "",
        "## Best Temporal-Holdout Model",
        "",
    ]
    if best.empty:
        lines.append("No valid temporal-holdout model was produced.")
    else:
        row = best.iloc[0]
        lines.extend(
            [
                f"- Target: {row['target']}",
                f"- Model: {row['model']}",
                f"- Accuracy: {row['accuracy']:.3f}",
                f"- Balanced accuracy: {row['balanced_accuracy']:.3f}",
                f"- ROC-AUC: {row['roc_auc']:.3f}",
                f"- Average precision: {row['average_precision']:.3f}",
                f"- F1: {row['f1']:.3f}",
                f"- Recall: {row['recall']:.3f}",
                f"- Specificity: {row['specificity']:.3f}",
                f"- Brier score: {row['brier']:.3f}",
                "",
            ]
        )
    if not ci.empty:
        lines.extend(["## Bootstrap 95% Confidence Intervals", "", ci.to_markdown(index=False), ""])
    lines.extend(
        [
            "## Reviewer-Safe Interpretation",
            "",
            "The model should be described as an early-warning classifier for schedule-risk prioritization, not as a causal model of weather delay. Weather and storm variables are exposure indicators. Schedule variance is defined from the public dashboard's forecast-completion changes and may reflect agency rebaselining, reporting updates, scope changes, procurement issues, or actual field conditions.",
            "",
            "## Key Verification Controls",
            "",
            "- No survey, image, video, or computer-vision-derived data are used.",
            "- The main target is shifted to the next reporting period to reduce same-row leakage.",
            "- The primary validation is a future-period holdout.",
            "- A project-group holdout is included to test whether performance depends on repeated records for the same project.",
            "- Baseline logistic regression is compared with tree-based ensemble models.",
        ]
    )
    (RESULTS / "schedule_risk_model_card.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = prepare_project_data()
    dataset_summary = summarize_dataset(df)
    dataset_summary.to_csv(RESULTS / "dataset_summary.csv", index=False)

    configs = [
        ExperimentConfig(30, "next_forecast_extension_gt_30_days"),
        ExperimentConfig(60, "next_forecast_extension_gt_60_days"),
        ExperimentConfig(90, "next_forecast_extension_gt_90_days"),
    ]
    result_frames = []
    pred_frames = []
    ci_frames = []
    for config in configs:
        results_df, pred_df, ci_df = run_experiment(df, config)
        result_frames.append(results_df)
        pred_frames.append(pred_df)
        if not ci_df.empty:
            ci_frames.append(ci_df)

    all_results = pd.concat(result_frames, ignore_index=True)
    all_predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    all_ci = pd.concat(ci_frames, ignore_index=True) if ci_frames else pd.DataFrame()
    all_results.to_csv(RESULTS / "model_results_all.csv", index=False)
    all_predictions.to_csv(RESULTS / "model_predictions_holdouts.csv", index=False)
    all_ci.to_csv(RESULTS / "best_model_bootstrap_ci.csv", index=False)

    best_by_target = (
        all_results[all_results["split"].eq("temporal_2025_2026_holdout")]
        .sort_values(["target", "roc_auc", "balanced_accuracy"], ascending=[True, False, False])
        .groupby("target", as_index=False)
        .head(1)
    )
    best_by_target.to_csv(RESULTS / "best_temporal_model_by_target.csv", index=False)
    write_model_card(all_results, all_ci, dataset_summary)

    manifest = {
        "raw_data_files": sorted(p.name for p in RAW.glob("*")),
        "processed_files": sorted(p.name for p in PROCESSED.glob("*")),
        "result_files": sorted(p.name for p in RESULTS.glob("*")),
        "nyc_schedule_url": NYC_SCHEDULE_URL,
        "nyc_budget_url": NYC_BUDGET_URL,
        "noaa_isd_station": "72503014732",
        "noaa_storm_directory": NOAA_STORM_DIR,
    }
    (RESULTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Completed schedule-risk study.")
    print(best_by_target[["target", "model", "split", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
