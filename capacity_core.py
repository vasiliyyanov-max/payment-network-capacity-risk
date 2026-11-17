"""capacity_core.py — общее ядро проекта Payment Network Capacity Risk & Early Warning (v7).

Здесь одна реализация всех расчётов, которыми пользуются и пайплайн, и дашборд:
эпизоды предупреждений, попадание в события, оценка порогов, экономика, бутстрап.
Поэтому цифры в отчётах и на экране совпадают.

Денежные суммы — условные единицы синтетической модели, не реальные данные банка или платёжной системы.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    config_version: str = "7.0.0"
    seed: int = 20251029
    start_ts: str = "2023-01-01 00:00:00"
    n_hours: int = 3 * 8760
    n_components: int = 6

    # физика очереди и определение события
    backlog_threshold: float = 220.0       # очередь выше порога = перегрузка (событие)
    min_event_hours: int = 2               # короче — не считается событием
    event_merge_gap_hours: int = 2         # разрывы короче — объединяются в одно событие

    # предупреждение
    horizon_hours: int = 6                 # окно: «событие начнётся в ближайшие N часов»
    alert_cooldown_hours: int = 3          # сигналы, разделённые тишиной короче, — один эпизод

    # временное разбиение (доли от n_hours) с purge и embargo
    train_frac: float = 0.55
    model_valid_frac: float = 0.10
    calib_frac: float = 0.10
    threshold_frac: float = 0.10
    embargo_hours: int = 12

    # экономика (условные единицы)
    missed_event_loss: float = 100_000.0   # ущерб от события без предупреждения
    review_cost: float = 2_500.0           # проверка аналитиком, на эпизод Review
    restrict_action_cost: float = 8_000.0  # переключение потока, на эпизод Restrict
    alert_hour_cost: float = 150.0         # мониторинг и разбор: за каждый час в режиме тревоги
    success_review: float = 0.55           # доля ущерба, предотвращённая предупреждением Review
    success_restrict: float = 0.75         # то же для Restrict

    # ограничение нагрузки на аналитиков при подборе порогов
    max_alert_episodes_per_day: float = 1.0
    max_alert_hour_share: float = 0.03     # не больше 3% часов в режиме тревоги

    n_bootstrap: int = 300


# ---------------------------------------------------------------------------
# События перегрузки
# ---------------------------------------------------------------------------

def find_breach_events(backlog, threshold: float, min_hours: int, merge_gap: int) -> list[tuple[int, int]]:
    """Непрерывные интервалы backlog > threshold, объединённые при разрывах <= merge_gap."""
    breach = np.asarray(backlog, dtype=float) > threshold
    runs: list[list[int]] = []
    i, n = 0, len(breach)
    while i < n:
        if breach[i]:
            s = i
            while i < n and breach[i]:
                i += 1
            runs.append([s, i - 1])
        else:
            i += 1
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and (s - merged[-1][1] - 1) <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s + 1 >= min_hours]


# ---------------------------------------------------------------------------
# Оценка политики порогов
# ---------------------------------------------------------------------------

def _episodes(fh: np.ndarray, fs: np.ndarray, cooldown: int):
    """Эпизоды предупреждений из отсортированных часов флагов.

    Возвращает первый и последний час эпизода, его максимальный уровень и число часов с флагом.
    """
    if len(fh) == 0:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([], dtype=np.int8), e
    new = np.r_[True, (np.diff(fh) - 1) > cooldown]
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:] - 1, len(fh) - 1]
    ep_start = fh[starts]
    ep_last = fh[ends]
    ep_level = np.maximum.reduceat(fs, starts)
    ep_hours = (ends - starts + 1).astype(np.int64)
    return ep_start, ep_last, ep_level, ep_hours


def reprice(ev_df: pd.DataFrame, ep_df: pd.DataFrame, cfg: Config) -> dict:
    """Экономика при заданных стоимостях. Эпизоды и попадания фиксированы, меняются только деньги."""
    lvl = ev_df["level"].to_numpy()
    success = np.where(lvl == 2, cfg.success_restrict, np.where(lvl == 1, cfg.success_review, 0.0))
    avoided = float((ev_df["caught"].to_numpy() * cfg.missed_event_loss * success).sum())
    is_r1 = (ep_df["level"] == 1).to_numpy()
    is_r2 = (ep_df["level"] == 2).to_numpy()
    hours = ep_df["n_hours"].to_numpy() if len(ep_df) else np.array([])
    review_total = float(is_r1.sum() * cfg.review_cost + (hours * is_r1).sum() * cfg.alert_hour_cost)
    restrict_total = float(is_r2.sum() * cfg.restrict_action_cost + (hours * is_r2).sum() * cfg.alert_hour_cost)
    cost = review_total + restrict_total
    baseline = len(ev_df) * cfg.missed_event_loss
    net = avoided - cost
    return {
        "baseline_loss": float(baseline),
        "avoided_loss": avoided,
        "review_cost": review_total,
        "restrict_cost": restrict_total,
        "alert_cost": float(cost),
        "net_benefit": float(net),
        "roi": float(net / cost) if cost > 0 else float("nan"),
    }


def policy_outcome(hours, score, onsets, cfg: Config, t_review: float, t_restrict: float, block: tuple[int, int]):
    """Оценка пары порогов на блоке.

    hours  — глобальные индексы часов, по которым есть прогноз (вне перегрузки)
    score  — калиброванный риск на этих часах
    onsets — глобальные индексы начала всех событий
    block  — (lo, hi) границы блока в часах; события берутся из [lo + H, hi)

    Возвращает (метрики, таблица событий, таблица эпизодов).
    """
    hours = np.asarray(hours, dtype=np.int64)
    score = np.asarray(score, dtype=float)
    order = np.argsort(hours, kind="stable")
    hours, score = hours[order], score[order]

    H = cfg.horizon_hours
    lo, hi = int(block[0]), int(block[1])
    onsets = np.sort(np.asarray(onsets, dtype=np.int64))
    ev = onsets[(onsets >= lo + H) & (onsets < hi)]

    state = np.where(score >= t_restrict, 2, np.where(score >= t_review, 1, 0)).astype(np.int8)
    flag = state > 0
    fh, fs = hours[flag], state[flag]

    # эпизоды предупреждений и их полезность
    ep_start, ep_last, ep_level, ep_hours = _episodes(fh, fs, cfg.alert_cooldown_hours)
    # полезен, если событие начинается после первого часа эпизода и не позже последнего часа + горизонт
    ep_useful = (np.searchsorted(ev, ep_last + H, "right") - np.searchsorted(ev, ep_start, "right")) > 0

    # попадание: есть ли тревога в окне [onset - H, onset - 1]
    n_ev = len(ev)
    caught = np.zeros(n_ev, dtype=int)
    lead = np.full(n_ev, np.nan)
    lvl = np.zeros(n_ev, dtype=np.int8)
    for k, o in enumerate(ev):
        i = np.searchsorted(fh, o - H, "left")
        j = np.searchsorted(fh, o - 1, "right")
        if j > i:
            caught[k] = 1
            lead[k] = o - fh[i]
            lvl[k] = fs[i:j].max()

    ev_df = pd.DataFrame({
        "onset_hour": ev, "caught": caught, "lead_h": lead, "level": lvl,
    })
    ep_df = pd.DataFrame({
        "start_hour": ep_start, "last_hour": ep_last, "level": ep_level.astype(np.int8),
        "n_hours": ep_hours, "useful": ep_useful.astype(int),
    })
    fixed = np.where(ep_df["level"] == 2, cfg.restrict_action_cost, cfg.review_cost) if len(ep_df) else np.array([])
    ep_df["cost"] = fixed + ep_df["n_hours"].to_numpy() * cfg.alert_hour_cost if len(ep_df) else np.array([])
    ev_df["avoided"] = ev_df["caught"] * cfg.missed_event_loss * np.where(
        ev_df["level"] == 2, cfg.success_restrict, np.where(ev_df["level"] == 1, cfg.success_review, 0.0))

    rp = reprice(ev_df, ep_df, cfg)
    days = (hi - lo) / 24.0
    n_caught = int(caught.sum())
    m = {
        "n_events": n_ev,
        "n_caught": n_caught,
        "recall": n_caught / n_ev if n_ev else float("nan"),
        "median_lead_h": float(np.median(lead[caught == 1])) if n_caught else float("nan"),
        "n_episodes": int(len(ep_df)),
        "precision": float(ep_df["useful"].mean()) if len(ep_df) else float("nan"),
        "alerts_per_day": len(ep_df) / days if days > 0 else float("nan"),
        "alert_hour_share": len(fh) / len(hours) if len(hours) else float("nan"),
        "restrict_episodes": int((ep_df["level"] == 2).sum()) if len(ep_df) else 0,
        "review_episodes": int((ep_df["level"] == 1).sum()) if len(ep_df) else 0,
        "days": days,
        **rp,
    }
    return m, ev_df, ep_df


def search_thresholds(hours, score, onsets, cfg: Config, block: tuple[int, int], n_quantiles: int = 60):
    """Перебор пар порогов по квантилям скора на блоке порогов.

    Цель — максимум чистого эффекта при двух ограничениях: эпизодов в сутки и доля часов в тревоге.
    Возвращает (лучшая строка, полная сетка, есть ли допустимые варианты).
    """
    qs = np.unique(np.quantile(np.asarray(score, dtype=float), np.linspace(0.60, 0.999, n_quantiles)))
    rows = []
    for t_rev in qs:
        for t_res in qs[qs > t_rev]:
            m, _, _ = policy_outcome(hours, score, onsets, cfg, t_rev, t_res, block)
            rows.append({
                "t_review": float(t_rev), "t_restrict": float(t_res),
                "net_benefit": m["net_benefit"], "recall": m["recall"], "precision": m["precision"],
                "alerts_per_day": m["alerts_per_day"], "alert_hour_share": m["alert_hour_share"],
                "n_episodes": m["n_episodes"],
                "median_lead_h": m["median_lead_h"], "roi": m["roi"],
            })
    grid = pd.DataFrame(rows)
    ok = (grid["alerts_per_day"] <= cfg.max_alert_episodes_per_day) & \
         (grid["alert_hour_share"] <= cfg.max_alert_hour_share)
    feasible = grid[ok]
    pool = feasible if len(feasible) else grid
    best = pool.loc[pool["net_benefit"].idxmax()]
    return best, grid, bool(len(feasible))


def precision_by_level(ep_df: pd.DataFrame, cfg: Config) -> dict:
    """Фактическая полезность эпизодов по уровням и точка безубыточности по полезности.

    Безубыточная полезность эпизода: cost / (missed_loss * success). Приближение: один эпизод
    отвечает за одно событие с тем же уровнем реагирования.
    """
    out = {}
    for lvl, name, cost, success in (
        (1, "review", cfg.review_cost, cfg.success_review),
        (2, "restrict", cfg.restrict_action_cost, cfg.success_restrict),
    ):
        s = ep_df[ep_df["level"] == lvl] if len(ep_df) else ep_df
        out[name] = {
            "episodes": int(len(s)),
            "precision": float(s["useful"].mean()) if len(s) else float("nan"),
            "breakeven_precision": cost / (cfg.missed_event_loss * success),
        }
    return out


def economics_steps(r: dict) -> pd.DataFrame:
    """Шаги для водопада: потери без системы → предотвращено → затраты → итог с системой."""
    residual = r["baseline_loss"] - r["avoided_loss"]
    total_with_system = residual + r["alert_cost"]
    return pd.DataFrame([
        {"step": "baseline", "label": "Потери без системы", "value": r["baseline_loss"], "kind": "total"},
        {"step": "avoided", "label": "Предотвращённые потери", "value": -r["avoided_loss"], "kind": "relative"},
        {"step": "review", "label": "Проверки (Review)", "value": r["review_cost"], "kind": "relative"},
        {"step": "restrict", "label": "Ограничения (Restrict)", "value": r["restrict_cost"], "kind": "relative"},
        {"step": "total", "label": "Итог с системой", "value": total_with_system, "kind": "total"},
    ])


def bootstrap_ci(ev_df: pd.DataFrame, ep_df: pd.DataFrame, cfg: Config, n_boot: int, seed: int) -> dict:
    """Стратифицированный бутстрап: отдельно пересэмплируем события и эпизоды.

    Для малых выборок это грубое приближение; интервалы честно широкие.
    """
    rng = np.random.default_rng(seed)
    n_e, n_p = len(ev_df), len(ep_df)
    keys = ("recall", "median_lead_h", "precision", "net_benefit", "roi")
    out = {k: [] for k in keys}
    if n_e == 0 or n_p == 0:
        return {k: {"low": float("nan"), "high": float("nan"), "mean": float("nan")} for k in keys}
    caught = ev_df["caught"].to_numpy()
    lead = ev_df["lead_h"].to_numpy()
    avoided = ev_df["avoided"].to_numpy()
    useful = ep_df["useful"].to_numpy()
    cost = ep_df["cost"].to_numpy()
    for _ in range(n_boot):
        ie = rng.integers(0, n_e, n_e)
        ip = rng.integers(0, n_p, n_p)
        c = caught[ie]
        out["recall"].append(c.mean())
        ld = lead[ie][c == 1]
        out["median_lead_h"].append(np.median(ld) if len(ld) else np.nan)
        out["precision"].append(useful[ip].mean())
        av = avoided[ie].mean() * n_e
        co = cost[ip].mean() * n_p
        out["net_benefit"].append(av - co)
        out["roi"].append((av - co) / co if co > 0 else np.nan)
    res = {}
    for k in keys:
        arr = np.asarray(out[k], dtype=float)
        res[k] = {
            "low": float(np.nanpercentile(arr, 2.5)),
            "high": float(np.nanpercentile(arr, 97.5)),
            "mean": float(np.nanmean(arr)),
        }
    return res


# ---------------------------------------------------------------------------
# Сценарий решений: три политики на одном стресс-окне
# ---------------------------------------------------------------------------

def scenario_policies(hourly: pd.DataFrame, events: pd.DataFrame, cfg: Config,
                      t_review: float, t_restrict: float, block: tuple[int, int]):
    """Сравнение политик на одном и том же сценарии.

    hourly — колонки component_id, hour, backlog_truth, score (по всем каналам и часам окна)
    events — колонки component_id, onset_hour (все события канала)
    Политики: без системы; только Review; Review + Restrict (с заданными порогами).

    Открытые эпизоды (их окно исхода не закончилось к концу блока) в затраты и полезность не входят:
    их исход неизвестен. Событие, которое уже произошло в блоке, учитывается по обычным правилам.
    Возвращает (таблица экономики по политикам, таблица эпизодов двухуровневой политики с признаком open).
    """
    lo, hi = int(block[0]), int(block[1])
    H = cfg.horizon_hours
    per_policy = {}
    for name, tr, tres in (("review_only", t_review, 1.01), ("two_tier", t_review, t_restrict)):
        evs, eps = [], []
        for comp, g in hourly.groupby("component_id"):
            g = g[(g["hour"] >= lo) & (g["hour"] < hi)]
            ok = g["backlog_truth"].to_numpy() <= cfg.backlog_threshold
            onset = events.loc[events["component_id"] == comp, "onset_hour"].to_numpy()
            _, e, p = policy_outcome(g["hour"].to_numpy()[ok], g["score"].to_numpy()[ok], onset,
                                     cfg, tr, tres, block)
            evs.append(e.assign(component_id=comp))
            eps.append(p.assign(component_id=comp))
        ev_all = pd.concat(evs, ignore_index=True)
        ep_all = pd.concat(eps, ignore_index=True)
        ep_all["open"] = (ep_all["last_hour"] + H >= hi).astype(bool) if len(ep_all) else pd.Series(dtype=bool)
        per_policy[name] = (ev_all, ep_all)

    ev_two, ep_two = per_policy["two_tier"]
    closed_two = ep_two[~ep_two["open"]] if len(ep_two) else ep_two
    rows = []
    no_ev = ev_two.assign(caught=0, avoided=0.0, level=0)
    r0 = reprice(no_ev, closed_two.iloc[0:0], cfg)
    rows.append({"policy": "no_system", "events": len(no_ev), "caught": 0, "recall": 0.0,
                 "episodes": 0, "open_episodes": 0, "precision": float("nan"), **r0})
    for name in ("review_only", "two_tier"):
        ev, ep_all = per_policy[name]
        closed = ep_all[~ep_all["open"]] if len(ep_all) else ep_all
        r = reprice(ev, closed, cfg)
        rows.append({"policy": name, "events": len(ev), "caught": int(ev["caught"].sum()),
                     "recall": float(ev["caught"].mean()) if len(ev) else float("nan"),
                     "episodes": len(closed),
                     "open_episodes": int(ep_all["open"].sum()) if len(ep_all) else 0,
                     "precision": float(closed["useful"].mean()) if len(closed) else float("nan"), **r})
    return pd.DataFrame(rows), ep_two
