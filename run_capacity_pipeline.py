"""run_capacity_pipeline.py — пайплайн Payment Network Capacity Risk & Early Warning (v7).

Порядок работы:
  1. Симуляция шести компонентов сети (спрос, ёмкость, очередь). Стресс-сценарии задаются
     как предвестники (нарастание нагрузки) и полные фазы; перегрузка возникает из динамики очереди,
     а не задаётся вручную. Ложные всплески без перегрузки тоже генерируются.
  2. Телеметрия с дефектами (пропуски, задержки, искажения).
  3. События = интервалы очереди выше порога (по истинной очереди). Метки: событие начнётся в ближайшие H часов.
  4. Временные блоки с purge и embargo.
  5. Три модели (константа без навыка, эвристика по очереди, логистическая регрессия, градиентный бустинг),
     Platt-калибровка на отдельном блоке.
  6. Пороги Review/Restrict подбираются на блоке порогов по чистому эффекту; тест не используется для подбора.
  7. Итоговая модель выбирается тоже по блоку порогов, оценка и бутстрап — на тесте.

Все числа выгружаются из расчёта. Ничего не задаётся вручную в выходных файлах.
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import capacity_core as core
from capacity_core import Config

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("capacity_pipeline_v7")

# тип стресса: (длина предвестника, часы полного эффекта, множитель спроса, множитель ёмкости)
STRESS_TYPES = {
    "gradual": (14, 22, 1.30, 0.82),   # постепенное нарастание нагрузки
    "joint": (6, 18, 1.32, 0.82),      # каскадный сбой с короткой предвестниковой фазой
    "flash": (0, 8, 1.55, 0.80),       # внезапный всплеск без предвестника
    "transient": (3, 5, 1.25, 1.00),   # кратковременный всплеск спроса, перегрузки обычно нет
    "precursor_only": (14, 0, 1.20, 0.92),  # только нарастание нагрузки, без полной фазы (для сценария)
}
TYPE_PROB = {"gradual": 0.30, "joint": 0.20, "flash": 0.20, "transient": 0.30}
FEATURE_BASES = ("demand", "capacity", "backlog", "util")


# ---------------------------------------------------------------------------
# 1. Симуляция
# ---------------------------------------------------------------------------

def draw_onsets(n: int, rng: np.random.Generator) -> list[tuple[int, str]]:
    """Случайный график стрессов одного канала. Порядок вызовов rng сохранён для воспроизводимости."""
    names = list(TYPE_PROB)  # случайные стрессы — только из TYPE_PROB; сценарные типы в основную симуляцию не попадают
    probs = np.array([TYPE_PROB[k] for k in names])
    onsets = []
    t = int(rng.exponential(700))
    while t < n - 60:
        typ = names[int(rng.choice(len(names), p=probs))]
        pre, full, _, _ = STRESS_TYPES[typ]
        onsets.append((t, typ))
        t += full + max(48, int(rng.exponential(700)))
    return onsets


def stress_arrays(n: int, onsets: list[tuple[int, str]]):
    """Мультипликаторы спроса и ёмкости, а также метка стресса по часам."""
    d_mult = np.ones(n)
    c_mult = np.ones(n)
    stress = np.array(["normal"] * n, dtype=object)
    for t, typ in onsets:
        pre, full, dm, cm = STRESS_TYPES[typ]
        if pre:
            idx = np.arange(max(t - pre, 0), min(t, n))  # часы за границей ряда не учитываются
            ramp = (idx - (t - pre) + 1) / (pre + 1)
            d_mult[idx] *= 1 + (dm - 1) * ramp
            c_mult[idx] *= 1 + (cm - 1) * ramp
        full_idx = np.arange(t, min(t + full, n))
        d_mult[full_idx] *= dm
        c_mult[full_idx] *= cm
        stress[max(t - pre, 0):min(t + full, n)] = typ
    return d_mult, c_mult, stress


def queue_path(n: int, b: float, phase: float, rng: np.random.Generator, d_mult, c_mult):
    """Спрос, ёмкость и истинная очередь канала."""
    hours = np.arange(n)
    daily = 0.10 * np.sin(2 * np.pi * (hours % 24) / 24 + phase)
    weekly = 0.05 * np.sin(2 * np.pi * (hours % 168) / 168 + phase)
    demand = b * (1 + daily + weekly) * d_mult + rng.normal(0, b * 0.035, n)
    capacity = b * 1.23 * (1 + 0.035 * np.sin(2 * np.pi * (hours % 24) / 24 + phase)) * c_mult \
        + rng.normal(0, b * 0.018, n)
    demand, capacity = np.clip(demand, 0, None), np.clip(capacity, 0, None)
    backlog = np.empty(n)
    q = 0.0
    for h in range(n):
        q = max(0.0, q + demand[h] - capacity[h])
        backlog[h] = q
    return demand, capacity, backlog


def simulate(cfg: Config, rng: np.random.Generator):
    n = cfg.n_hours
    hours = np.arange(n)
    ts = pd.date_range(cfg.start_ts, periods=n, freq="h", tz="UTC")
    base = np.linspace(760.0, 1180.0, cfg.n_components)

    frames, onset_rows = [], []
    for c in range(cfg.n_components):
        comp = f"COMP-{c + 1:02d}"
        onsets = draw_onsets(n, rng)
        for t, typ in onsets:
            onset_rows.append({"component_id": comp, "stress_onset_hour": t, "stress_type": typ})
        d_mult, c_mult, stress = stress_arrays(n, onsets)
        demand, capacity, backlog = queue_path(n, base[c], c * 0.7, rng, d_mult, c_mult)
        frames.append(pd.DataFrame({
            "timestamp_utc": ts, "hour": hours, "component_id": comp,
            "demand": demand, "capacity": capacity, "backlog_truth": backlog, "stress_type": stress,
        }))

    raw = pd.concat(frames, ignore_index=True)
    return raw, pd.DataFrame(onset_rows)


def make_observed(raw: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Телеметрия с дефектами: 0.25% строк — пропуск, задержка или искажение одного поля."""
    obs = raw.copy()
    obs["data_quality_status"] = "Data OK"
    obs["dq_defect_type"] = None
    obs["demand_obs"] = obs["demand"]
    obs["capacity_obs"] = obs["capacity"]
    obs["backlog_obs"] = obs["backlog_truth"]
    truth_col = {"demand_obs": "demand", "capacity_obs": "capacity", "backlog_obs": "backlog_truth"}

    n_def = int(len(obs) * 0.0025)
    for i in rng.choice(len(obs), size=n_def, replace=False):
        kind = str(rng.choice(["missing", "delayed", "distorted"]))
        field = str(rng.choice(list(truth_col)))
        obs.at[i, "data_quality_status"] = "Data issue"
        obs.at[i, "dq_defect_type"] = kind
        if kind == "missing":
            obs.at[i, field] = np.nan
        elif kind == "delayed" and i > 0 and obs.at[i - 1, "component_id"] == obs.at[i, "component_id"]:
            obs.at[i, field] = raw.at[i - 1, truth_col[field]]
        else:
            obs.at[i, field] = obs.at[i, field] * rng.uniform(0.7, 1.3)
    return obs


def build_events(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """События = интервалы перегрузки по истинной очереди, с указанием сценария-причины."""
    rows = []
    for comp, g in raw.groupby("component_id", sort=False):
        backlog = g["backlog_truth"].to_numpy()
        hours = g["hour"].to_numpy()
        stress = g["stress_type"].to_numpy()
        for s, e in core.find_breach_events(backlog, cfg.backlog_threshold, cfg.min_event_hours, cfg.event_merge_gap_hours):
            prior = [x for x in stress[max(0, s - 24):s + 1] if x != "normal"]
            rows.append({
                "component_id": comp,
                "onset_hour": int(hours[s]),
                "onset_ts": pd.Timestamp(cfg.start_ts, tz="UTC") + pd.Timedelta(hours=int(hours[s])),
                "duration_hours": int(e - s + 1),
                "peak_backlog": float(np.max(backlog[s:e + 1])),
                "scenario": prior[-1] if prior else "endogenous",
            })
    cat = pd.DataFrame(rows).sort_values("onset_hour").reset_index(drop=True)
    cat.insert(0, "event_id", [f"EV-{i + 1:04d}" for i in range(len(cat))])
    return cat


# ---------------------------------------------------------------------------
# 2. Признаки и метки
# ---------------------------------------------------------------------------

def build_features(obs: pd.DataFrame):
    df = obs.drop(columns=["demand", "capacity"])  # истинные значения в признаки не попадают
    df = df.rename(columns={"demand_obs": "demand", "capacity_obs": "capacity", "backlog_obs": "backlog"})
    df = df.sort_values(["component_id", "hour"]).reset_index(drop=True)
    df["util"] = df["demand"] / df["capacity"].clip(lower=1)
    g = df.groupby("component_id", group_keys=False)
    feats: list[str] = []

    def add(name: str, series) -> None:
        df[name] = series
        feats.append(name)

    for col in FEATURE_BASES:
        add(f"{col}_now", df[col])
        add(f"{col}_lag1", g[col].shift(1))
        add(f"{col}_d3h", g[col].shift(1) - g[col].shift(4))
        for w in (6, 12, 24):
            add(f"{col}_roll{w}", g[col].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=max(2, w // 2)).mean()))
    add("backlog_slope6", g["backlog"].shift(1) - g["backlog"].shift(7))
    add("backlog_slope12", g["backlog"].shift(1) - g["backlog"].shift(13))
    add("clearance_h", df["backlog"] / (df["capacity"] - df["demand"]).clip(lower=1))
    add("dq_flag", (df["data_quality_status"] != "Data OK").astype(int))
    add("hour_of_day", df["timestamp_utc"].dt.hour)
    add("day_of_week", df["timestamp_utc"].dt.dayofweek)
    return df, feats


def add_labels(df: pd.DataFrame, events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """y = 1, если событие начнётся в (t, t+H]. Часы внутри перегрузки исключаются из оценки."""
    H = cfg.horizon_hours
    hrs = df["hour"].to_numpy()
    y = np.full(len(df), np.nan)
    for comp, idx in df.groupby("component_id").indices.items():
        on = np.sort(events.loc[events["component_id"] == comp, "onset_hour"].to_numpy())
        h = hrs[idx]
        cnt = np.searchsorted(on, h + H, "right") - np.searchsorted(on, h, "right")
        y[idx] = (cnt > 0).astype(float)
    ok = df["backlog_truth"].to_numpy() <= cfg.backlog_threshold
    valid = ok & (hrs + H <= hrs.max())
    df["eval_ok"] = ok
    df["y"] = np.where(valid, y, np.nan)
    return df


def block_bounds(cfg: Config) -> dict[str, tuple[int, int]]:
    n, g = cfg.n_hours, cfg.embargo_hours
    tr = int(n * cfg.train_frac)
    mv = int(n * cfg.model_valid_frac)
    ca = int(n * cfg.calib_frac)
    th = int(n * cfg.threshold_frac)
    out = {"train": (0, tr)}
    p = tr
    out["validation_model"] = (p + g, p + g + mv); p = p + g + mv
    out["validation_calibration"] = (p + g, p + g + ca); p = p + g + ca
    out["validation_threshold"] = (p + g, p + g + th); p = p + g + th
    out["test"] = (p + g, n)
    return out


def block_masks(df: pd.DataFrame, blocks: dict, H: int) -> dict:
    h = df["hour"].to_numpy()
    lab_ok = df["y"].notna().to_numpy()
    eval_ok = df["eval_ok"].to_numpy()
    out = {}
    for name, (lo, hi) in blocks.items():
        inb = (h >= lo) & (h < hi)
        out[name] = {
            "eval": inb & eval_ok,
            "lab": inb & lab_ok & (h + H < hi),   # purge: метка целиком внутри блока
        }
    return out


# ---------------------------------------------------------------------------
# 3. Модели и калибровка
# ---------------------------------------------------------------------------

def logit(p):
    p = np.clip(np.nan_to_num(p, nan=0.0), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(raw: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1e4, max_iter=1000).fit(logit(raw).reshape(-1, 1), y)


def apply_platt(lr: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    return lr.predict_proba(logit(raw).reshape(-1, 1))[:, 1]


def model_suite(seed: int) -> dict:
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000)),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, class_weight="balanced", random_state=seed),
    }


def stat_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    prev = float(y.mean())
    brier = float(brier_score_loss(y, p))
    ref = prev * (1 - prev)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": brier,
        "brier_skill": 1 - brier / ref if ref > 0 else float("nan"),
        "prevalence": prev,
        "n_labelled": int(len(y)),
        "n_positive": int(y.sum()),
    }


def compute_psi(a: np.ndarray, b: np.ndarray, buckets: int = 10) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    edges = np.percentile(a, np.linspace(0, 100, buckets + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    pa = np.clip(np.histogram(a, edges)[0] / len(a), 1e-4, None)
    pb = np.clip(np.histogram(b, edges)[0] / len(b), 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


# ---------------------------------------------------------------------------
# 4a. Сценарий решений: заданный стресс в выбранных каналах
# ---------------------------------------------------------------------------
# Стресс задаётся как входные данные (тип и момент начала, часы от начала окна).
# Состояния Review / Restrict НЕ задаются: они появляются из выхода обученной модели
# и порогов, подобранных на блоке порогов. Каналы без стресса в сценарии остаются штатными.
SCEN_WARM_H = 96            # прогрев: 4 суток штатной работы, без оценки
SCEN_LEN_H = 24 * 14        # окно сценария: 14 суток
SCENARIO_SCHEDULE: dict[str, list[tuple[int, str]]] = {
    "COMP-06": [(120, "precursor_only"),     # Review без перегрузки: закрытая ложная тревога
                (340, "precursor_only")],    # Review в конце окна: текущее состояние, исход ещё неизвестен
    "COMP-04": [(300, "gradual")],           # ожидаемо: Review, затем Restrict до перегрузки
}
ACTION_CODES = {"Review": "review_channel", "Restrict": "switch_flow"}


def run_scenario(cfg: Config, models: dict, platt: dict, chosen: str, t_rev: float, t_res: float,
                 schedule: dict, out_dir: Path, seed: int) -> dict:
    n = SCEN_WARM_H + SCEN_LEN_H
    start = cfg.n_hours
    rng = np.random.default_rng(seed)
    base = np.linspace(760.0, 1180.0, cfg.n_components)
    ts0 = pd.Timestamp(cfg.start_ts, tz="UTC")

    frames = []
    for c in range(cfg.n_components):
        comp = f"COMP-{c + 1:02d}"
        onsets = [(SCEN_WARM_H + off, typ) for off, typ in schedule.get(comp, [])]
        d_mult, c_mult, stress = stress_arrays(n, onsets)
        demand, capacity, backlog = queue_path(n, base[c], c * 0.7, rng, d_mult, c_mult)
        hrs = start + np.arange(n)
        frames.append(pd.DataFrame({
            "timestamp_utc": [ts0 + pd.Timedelta(hours=int(h)) for h in hrs],
            "hour": hrs, "component_id": comp,
            "demand": demand, "capacity": capacity, "backlog_truth": backlog, "stress_type": stress,
            "demand_obs": demand, "capacity_obs": capacity, "backlog_obs": backlog,
            "data_quality_status": "Data OK", "dq_defect_type": None,
        }))
    raw = pd.concat(frames, ignore_index=True)

    df, feats = build_features(raw)
    X = df[feats].to_numpy(dtype=float)
    if chosen == "rule_backlog_level":
        raw_p = np.clip(np.nan_to_num(df["backlog_now"].to_numpy() / cfg.backlog_threshold), 0, 1)
    else:
        raw_p = models[chosen].predict_proba(X)[:, 1]
    df["score"] = apply_platt(platt[chosen], raw_p)
    df["eval_ok"] = df["backlog_truth"].to_numpy() <= cfg.backlog_threshold
    block = (start + SCEN_WARM_H, start + n)
    events = build_events(raw, cfg)
    in_win = df["hour"].to_numpy() >= block[0]

    econ, ep_two = core.scenario_policies(df.loc[:, ["component_id", "hour", "backlog_truth", "score"]],
                                          events, cfg, t_rev, t_res, block)
    econ.to_csv(out_dir / "scenario_economics.csv", index=False)

    ts0s = ts0
    ep = ep_two.copy()
    ep["start_ts"] = [ts0s + pd.Timedelta(hours=int(h)) for h in ep["start_hour"]]
    ep["level_name"] = ep["level"].map({1: "Review", 2: "Restrict"})
    ep["action_code"] = ep["level_name"].map(ACTION_CODES)
    ep["start_ts"] = ep["start_ts"].astype(str)
    ep.to_csv(out_dir / "scenario_episodes.csv", index=False)

    evt = events.copy()
    evt["onset_ts"] = evt["onset_ts"].astype(str)
    evt["in_scenario"] = (evt["onset_hour"] >= block[0] + cfg.horizon_hours) & (evt["onset_hour"] < block[1])
    evt.to_csv(out_dir / "scenario_events.csv", index=False)

    hourly = df.loc[in_win, ["timestamp_utc", "hour", "component_id", "demand", "capacity",
                             "backlog_truth", "score", "eval_ok"]].copy()
    hourly["state"] = np.where(~hourly["eval_ok"].astype(bool), "Breach",
                               np.where(hourly["score"] >= t_res, "Restrict",
                                        np.where(hourly["score"] >= t_rev, "Review", "Normal")))
    hourly["action_code"] = hourly["state"].map(ACTION_CODES)
    hourly["timestamp_utc"] = hourly["timestamp_utc"].astype(str)
    hourly.drop(columns=["eval_ok"]).to_csv(out_dir / "scenario_hourly.csv", index=False)

    meta = {
        "warm_hours": SCEN_WARM_H, "window_hours": SCEN_LEN_H, "block": [int(block[0]), int(block[1])],
        "window_start_ts": str(ts0 + pd.Timedelta(hours=int(block[0]))),
        "schedule": {k: [{"offset_h": o, "type": t} for o, t in v] for k, v in schedule.items()},
        "max_score_by_component": {c: float(g["score"].max()) for c, g in hourly.groupby("component_id")},
        "state_counts": hourly["state"].value_counts().to_dict(),
        "thresholds": {"review": t_rev, "restrict": t_res},
    }
    (out_dir / "scenario_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# 4. Полный прогон
# ---------------------------------------------------------------------------

def run(out_dir: Path, cfg: Config = Config()) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    H = cfg.horizon_hours

    LOG.info("1/6 Симуляция сети (%d компонентов, %d часов)", cfg.n_components, cfg.n_hours)
    raw, stress_df = simulate(cfg, rng)
    obs = make_observed(raw, rng)

    LOG.info("2/6 События перегрузки по истинной очереди")
    events = build_events(raw, cfg)

    LOG.info("3/6 Признаки, метки, временные блоки")
    df, feats = build_features(obs)
    df = add_labels(df, events, cfg)
    blocks = block_bounds(cfg)
    M = block_masks(df, blocks, H)
    hour = df["hour"].to_numpy()
    y = df["y"].to_numpy()
    X = df[feats].to_numpy(dtype=float)
    onsets = events["onset_hour"].to_numpy()

    LOG.info("4/6 Обучение моделей и Platt-калибровка")
    raw_scores: dict[str, np.ndarray] = {}
    fitted = model_suite(cfg.seed)
    tr = M["train"]["lab"]
    for name, mdl in fitted.items():
        mdl.fit(X[tr], y[tr].astype(int))
        raw_scores[name] = mdl.predict_proba(X)[:, 1]
    raw_scores["rule_backlog_level"] = np.clip(np.nan_to_num(df["backlog_now"].to_numpy() / cfg.backlog_threshold), 0, 1)

    cal = M["validation_calibration"]["lab"]
    cal_scores, platt_fit = {}, {}
    for name, raw_p in raw_scores.items():
        platt_fit[name] = fit_platt(raw_p[cal], y[cal].astype(int))
        cal_scores[name] = apply_platt(platt_fit[name], raw_p)

    LOG.info("5/6 Пороги на блоке порогов, оценка на тесте")
    tb = blocks["validation_threshold"]
    te_thr = M["validation_threshold"]["eval"]
    tb_test = blocks["test"]
    te_eval = M["test"]["eval"]
    te_lab = M["test"]["lab"]

    comparison, frozen_by_model, grids = [], {}, {}
    for name, s in cal_scores.items():
        best, grid, feasible = core.search_thresholds(hour[te_thr], s[te_thr], onsets, cfg, tb)
        thr_net = float(best["net_benefit"])
        m_test, ev_t, ep_t = core.policy_outcome(
            hour[te_eval], s[te_eval], onsets, cfg, best["t_review"], best["t_restrict"], tb_test)
        st = stat_metrics(y[te_lab].astype(int), s[te_lab])
        comparison.append({
            "model": name,
            "roc_auc_test": st["roc_auc"], "pr_auc_test": st["pr_auc"],
            "brier_test": st["brier"], "brier_skill_test": st["brier_skill"],
            "net_on_threshold_block": thr_net,
            "t_review": float(best["t_review"]), "t_restrict": float(best["t_restrict"]),
            "feasible_under_alert_limit": feasible,
            "recall_test": m_test["recall"], "median_lead_h_test": m_test["median_lead_h"],
            "precision_test": m_test["precision"], "alerts_per_day_test": m_test["alerts_per_day"],
            "net_benefit_test": m_test["net_benefit"], "roi_test": m_test["roi"],
        })
        frozen_by_model[name] = (best, m_test, ev_t, ep_t, st)
        grids[name] = grid

    cmp_df = pd.DataFrame(comparison)
    cmp_df["selected"] = False
    feas_mask = cmp_df["feasible_under_alert_limit"].astype(bool)
    sel_pool = cmp_df[feas_mask] if feas_mask.any() else cmp_df
    chosen = sel_pool.loc[sel_pool["net_on_threshold_block"].idxmax(), "model"]
    cmp_df.loc[cmp_df["model"] == chosen, "selected"] = True

    # прогноз без навыка: константа распространённости
    prev_train = float(y[tr].astype(int).mean())
    te_p = np.full(te_lab.sum(), prev_train)
    no_skill = stat_metrics(y[te_lab].astype(int), te_p)
    cmp_df = pd.concat([cmp_df, pd.DataFrame([{
        "model": "no_skill_constant", "roc_auc_test": 0.5, "pr_auc_test": no_skill["pr_auc"],
        "brier_test": no_skill["brier"], "brier_skill_test": 0.0, "selected": False,
    }])], ignore_index=True)

    best, m_test, ev_t, ep_t, st_chosen = frozen_by_model[chosen]
    s_chosen = cal_scores[chosen]
    t_rev, t_res = float(best["t_review"]), float(best["t_restrict"])

    LOG.info("6/6 Бутстрап, экономика, сценарии, дрейф, выгрузка")
    ci = core.bootstrap_ci(ev_t, ep_t, cfg, cfg.n_bootstrap, cfg.seed)
    econ = core.reprice(ev_t, ep_t, cfg)
    steps = core.economics_steps(econ)
    prec_lvl = core.precision_by_level(ep_t, cfg)

    scen_map = events.set_index("onset_hour")["scenario"]
    ev_t = ev_t.assign(scenario=ev_t["onset_hour"].map(scen_map))
    scenario_rows = []
    for scen, g in ev_t.groupby("scenario"):
        scenario_rows.append({
            "scenario": scen, "events": int(len(g)), "caught": int(g["caught"].sum()),
            "recall": float(g["caught"].mean()),
            "median_lead_h": float(g.loc[g["caught"] == 1, "lead_h"].median()) if g["caught"].sum() else float("nan"),
        })
    scenario_df = pd.DataFrame(scenario_rows)

    ts0 = pd.Timestamp(cfg.start_ts, tz="UTC")
    ev_t_out = ev_t.assign(onset_ts=[ts0 + pd.Timedelta(hours=int(h)) for h in ev_t["onset_hour"]])
    ep_t_out = ep_t.assign(start_ts=[ts0 + pd.Timedelta(hours=int(h)) for h in ep_t["start_hour"]],
                           level_name=ep_t["level"].map({1: "Review", 2: "Restrict"}))

    # калибровка на тесте
    p_t, y_t = s_chosen[te_lab], y[te_lab].astype(int)
    cal_df = pd.DataFrame({"p": p_t, "y": y_t})
    cal_df["bin"] = pd.qcut(cal_df["p"], 10, duplicates="drop")
    calib_points = cal_df.groupby("bin", observed=True).agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index(drop=True)

    # дрейф признаков: train → test
    tr_all = M["train"]["eval"]
    te_all = M["test"]["eval"]
    psi_rows = []
    for f in feats:
        v = compute_psi(df.loc[tr_all, f].to_numpy(dtype=float), df.loc[te_all, f].to_numpy(dtype=float))
        status = "Stable" if v < 0.1 else ("Moderate drift" if v < 0.25 else "Action needed")
        psi_rows.append({"feature": f, "psi": v, "status": status})
    psi_df = pd.DataFrame(psi_rows)

    # важность признаков градиентного бустинга на тесте
    gb = fitted["gradient_boosting"]
    pi = permutation_importance(gb, X[te_lab], y[te_lab].astype(int), n_repeats=3,
                                random_state=cfg.seed, scoring="average_precision")
    imp_df = pd.DataFrame({"feature": feats, "importance_mean": pi.importances_mean,
                           "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)

    # выгрузка: прогнозы на тесте (включая часы перегрузки для графиков)
    idx_t = np.flatnonzero((hour >= tb_test[0]) & (hour < tb_test[1]))
    tst = df.loc[idx_t, ["timestamp_utc", "hour", "component_id", "demand", "capacity", "backlog",
                         "backlog_truth", "data_quality_status", "y", "eval_ok"]].copy()
    tst = tst.rename(columns={"demand": "demand_obs", "capacity": "capacity_obs", "backlog": "backlog_obs"})
    for name in cal_scores:
        tst[f"score_{name}"] = cal_scores[name][idx_t]
    tst["risk_state"] = np.where(~tst["eval_ok"].astype(bool), "Breach",
                                 np.where(tst[f"score_{chosen}"] >= t_res, "Restrict",
                                          np.where(tst[f"score_{chosen}"] >= t_rev, "Review", "Normal")))
    tst["timestamp_utc"] = tst["timestamp_utc"].astype(str)
    tst.to_csv(out_dir / "test_predictions.csv", index=False)

    # выгрузка: остальные файлы
    events_out = events.copy()
    events_out["split"] = "none"
    for name, (lo, hi) in blocks.items():
        events_out.loc[(events_out["onset_hour"] >= lo) & (events_out["onset_hour"] < hi), "split"] = name
    events_out["onset_ts"] = events_out["onset_ts"].astype(str)
    events_out.to_csv(out_dir / "event_catalog.csv", index=False)
    ev_t_out.assign(onset_ts=ev_t_out["onset_ts"].astype(str)).to_csv(out_dir / "test_events.csv", index=False)
    ep_t_out.assign(start_ts=ep_t_out["start_ts"].astype(str)).to_csv(out_dir / "test_episodes.csv", index=False)
    cmp_df.to_csv(out_dir / "model_comparison.csv", index=False)
    grids[chosen].to_csv(out_dir / "threshold_grid.csv", index=False)
    steps.to_csv(out_dir / "economics_steps.csv", index=False)
    scenario_df.to_csv(out_dir / "scenario_breakdown.csv", index=False)
    calib_points.to_csv(out_dir / "calibration_points.csv", index=False)
    psi_df.to_csv(out_dir / "feature_psi_report.csv", index=False)
    imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
    stress_df.to_csv(out_dir / "stress_onsets.csv", index=False)

    dq = obs.loc[obs["data_quality_status"] != "Data OK", "dq_defect_type"].value_counts().to_dict()
    summary = {
        "hours": cfg.n_hours,
        "components": cfg.n_components,
        "start_ts": cfg.start_ts,
        "events_total": int(len(events)),
        "events_by_scenario": events["scenario"].value_counts().to_dict(),
        "stress_onsets_by_type": stress_df["stress_type"].value_counts().to_dict(),
        "dq_defects": {k: int(v) for k, v in dq.items()},
        "dq_defect_rows": int(sum(dq.values())),
        "blocks": {k: list(v) for k, v in blocks.items()},
        "prevalence_by_block": {k: float(y[M[k]["lab"]].astype(int).mean()) for k in blocks if M[k]["lab"].any()},
    }
    (out_dir / "data_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    frozen = {
        "config_version": cfg.config_version,
        "chosen_model": chosen,
        "thresholds": {"review": t_rev, "restrict": t_res},
        "threshold_search_feasible": bool(cmp_df.loc[cmp_df["model"] == chosen, "feasible_under_alert_limit"].iloc[0]),
        "policy_block": list(tb_test),
        "statistical_test": st_chosen,
        "event_metrics_test": {k: m_test[k] for k in (
            "n_events", "n_caught", "recall", "median_lead_h", "n_episodes", "precision",
            "alerts_per_day", "restrict_episodes", "review_episodes", "days")},
        "economics_test": econ,
        "precision_by_level": prec_lvl,
        "ci95_test": ci,
        "scenario_breakdown": scenario_rows,
        "breakeven_hourly_review": cfg.review_cost / (cfg.review_cost + cfg.missed_event_loss * cfg.success_review),
        "config": asdict(cfg),
    }
    (out_dir / "frozen_metrics.json").write_text(json.dumps(frozen, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    scen_meta = run_scenario(cfg, fitted, platt_fit, chosen, t_rev, t_res, SCENARIO_SCHEDULE,
                             out_dir, cfg.seed + 7)
    LOG.info("Сценарий: состояния %s", scen_meta["state_counts"])
    LOG.info("Готово: выбрана модель %s, пороги Review=%.4f Restrict=%.4f", chosen, t_rev, t_res)
    return frozen


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    run(here / "capacity_output")
