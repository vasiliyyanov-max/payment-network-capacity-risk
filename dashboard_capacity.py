"""dashboard_capacity.py — дашборд Payment Network Capacity Risk & Early Warning (v7).

Дашборд не содержит захардкоженных результатов. Все цифры берутся из файлов пайплайна
(run_capacity_pipeline.py) или пересчитываются ядром capacity_core при изменении параметров.

Запуск:
    python run_capacity_pipeline.py          # один раз, создаёт папку capacity_output
    streamlit run dashboard_capacity.py

Денежные суммы — условные единицы синтетической модели.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import capacity_core as core
from capacity_core import Config

st.set_page_config(page_title="Payment Network Capacity Risk", page_icon="🛰️", layout="wide")

DATA_DIR = Path(os.environ.get("CAPACITY_OUTPUT", Path(__file__).resolve().parent / "capacity_output"))
COL = {"Normal": "#16a34a", "Review": "#f59e0b", "Restrict": "#dc2626", "Breach": "#334155"}
NAVY = "#1e3a8a"
TEAL = "#0f766e"
GRID = "#e2e8f0"

# ---------------------------------------------------------------------------
# Тексты
# ---------------------------------------------------------------------------
SCN = {'ru': {'head': 'Сценарий решений: заданный стресс в двух каналах', 'note': 'Стресс задан во входных данных: COMP-06 — нарастание нагрузки без перегрузки, COMP-04 — нарастание, которое переходит в перегрузку. Состояния Review и Restrict возникают из обученной модели и порогов; здесь ничего не задано вручную. Каналы без стресса остаются штатными, редкие Review на них — ложные тревоги.', 'pols': 'Три политики на одном и том же сценарии', 'pol_no': 'Без системы', 'pol_review': 'Только Review', 'pol_two': 'Review + Restrict', 'col_policy': 'Политика', 'col_events': 'Событий', 'col_caught': 'С предупреждением', 'col_avoided': 'Предотвращено', 'col_costs': 'Затраты', 'col_net': 'Чистый эффект', 'col_roi': 'ROI', 'gain': 'Вклад уровня Restrict', 'gain_s': 'чистый эффект Review + Restrict минус только Review', 'chart_risk': 'Риск и пороги', 'chart_bl': 'Очередь и порог перегрузки', 'focus': 'Фокус сценария', 'log': 'Журнал решений по всем каналам сценария', 'log_t': 'Время', 'log_c': 'Канал', 'log_l': 'Уровень', 'log_a': 'Действие', 'log_h': 'Часов', 'log_r': 'Результат', 'useful': 'полезен', 'false': 'ложная тревога', 'act_r': 'Проверить резервный канал и подготовить перевод 20% потока', 'act_s': 'Переключить 30% потока на резервный процессор', 'missing': 'Файлы сценария не найдены. Запустите пайплайн v7 заново.'}, 'en': {'head': 'Decision scenario: designed stress in two channels', 'note': 'The stress is an input: COMP-06 is a load build-up without overload, COMP-04 is a build-up that turns into overload. The Review and Restrict states come from the trained model and thresholds; nothing is set by hand. Channels without stress stay normal; occasional Review there are false alarms.', 'pols': 'Three policies on the same scenario', 'pol_no': 'No system', 'pol_review': 'Review only', 'pol_two': 'Review + Restrict', 'col_policy': 'Policy', 'col_events': 'Events', 'col_caught': 'Warned', 'col_avoided': 'Avoided', 'col_costs': 'Costs', 'col_net': 'Net benefit', 'col_roi': 'ROI', 'gain': 'Value of the Restrict tier', 'gain_s': 'net benefit of Review + Restrict minus Review only', 'chart_risk': 'Risk and thresholds', 'chart_bl': 'Backlog and overload threshold', 'focus': 'Scenario focus', 'log': 'Decision log for all scenario channels', 'log_t': 'Time', 'log_c': 'Channel', 'log_l': 'Level', 'log_a': 'Action', 'log_h': 'Hours', 'log_r': 'Result', 'useful': 'useful', 'false': 'false alarm', 'act_r': 'Check the backup channel and prepare to move 20% of flow', 'act_s': 'Switch 30% of flow to the backup processor', 'missing': 'Scenario files not found. Re-run the v7 pipeline.'}, 'uk': {'head': 'Сценарій рішень: заданий стрес у двох каналах', 'note': 'Стрес задано у вхідних даних: COMP-06 — наростання навантаження без перевантаження, COMP-04 — наростання, яке переходить у перевантаження. Стани Review і Restrict виникають із навченої моделі та порогів; нічого не задано вручну. Канали без стресу лишаються штатними, рідкісні Review на них — хибні тривоги.', 'pols': 'Три політики на одному сценарії', 'pol_no': 'Без системи', 'pol_review': 'Лише Review', 'pol_two': 'Review + Restrict', 'col_policy': 'Політика', 'col_events': 'Подій', 'col_caught': 'З попередженням', 'col_avoided': 'Запобігнуто', 'col_costs': 'Витрати', 'col_net': 'Чистий ефект', 'col_roi': 'ROI', 'gain': 'Внесок рівня Restrict', 'gain_s': 'чистий ефект Review + Restrict мінус лише Review', 'chart_risk': 'Ризик і пороги', 'chart_bl': 'Черга і поріг перевантаження', 'focus': 'Фокус сценарію', 'log': 'Журнал рішень за всіма каналами сценарію', 'log_t': 'Час', 'log_c': 'Канал', 'log_l': 'Рівень', 'log_a': 'Дія', 'log_h': 'Годин', 'log_r': 'Результат', 'useful': 'корисний', 'false': 'хибна тривога', 'act_r': 'Перевірити резервний канал і підготувати переведення 20% потоку', 'act_s': 'Переключити 30% потоку на резервний процесор', 'missing': 'Файли сценарію не знайдено. Перезапустіть пайплайн v7.'}}

T = {
    "ru": {
        "infeasible": 'Для этой модели нет допустимых порогов: доля часов в режиме тревоги выше лимита 3%. Полезность и экономика не рассчитываются.', "infeasible_kpi": 'Показатели не рассчитываются: для этой модели нет допустимых порогов.', "no_thr": 'нет допустимых порогов', "col_share": 'Часов в тревоге',
        "model_sel": 'Модель для просмотра', "model_note": 'Переключение меняет обзор, канал, экономику и сценарий. Отметка «выбрана» в таблице — модель, выбранная по правилу.',
        "ops": 'Оперативный режим: последние часы сценария', "ops_now": 'на', "heat_leg": 'Нормально — тёмный · Review — жёлтый · Restrict — красный · Breach (перегрузка) — чёрный · × — начало события', "params": 'Параметры расчёта', "open_note": 'Открытых эпизодов (исход ещё не наступил): {n}. В затраты не включены.',
        "psi_range": 'Диапазон PSI по всем признакам', "feat_title": 'Что означают признаки', "feat_name": 'Признак', "feat_group": 'Группа', "feat_what": 'Что измеряет', "feat_note": '**PSI** сравнивает распределение признака на обучающем и тестовом периодах. Почти нулевые значения означают, что в симуляции нет структурных сдвигов. Признаки очереди (backlog) почти всегда равны нулю, поэтому их бины квантилей вырождены, и PSI для них формально нулевой. Это ограничение метода, а не признак стабильности.',
        "title": "Payment Network Capacity Risk & Early Warning",
        "subtitle": "Раннее предупреждение о перегрузке каналов P2P-платёжной сети",
        "synthetic": "Все данные синтетические. Результаты демонстрируют методологию и не являются доказательством эффективности на реальных данных.",
        "lang": "Язык",
        "thr_box": "Пороги решения",
        "use_frozen": "Замороженные пороги из пайплайна",
        "thr_review": "Порог Review",
        "thr_restrict": "Порог Restrict",
        "thr_warn": "Порог Restrict ниже порога Review — значение поднято до уровня Review.",
        "cost_box": "Экономика (условные единицы)",
        "missed": "Ущерб от события без предупреждения",
        "c_review": "Проверка (Review), на эпизод",
        "c_restrict": "Ограничение (Restrict), на эпизод",
        "c_hour": "Мониторинг, за час тревоги",
        "s_review": "Доля ущерба, предотвращённая Review",
        "s_restrict": "Доля ущерба, предотвращённая Restrict",
        "w_base": "Потери без системы", "w_avoid": "Предотвращённые потери", "w_review": "Проверки (Review)", "w_restrict": "Ограничения (Restrict)", "w_total": "Итог с системой",
        "tabs": ["Обзор", "Канал", "Модели", "Экономика", "Сценарий", "Данные и дрейф", "Методология"],
        "kpi_recall": "Полнота (recall)",
        "kpi_recall_s": "доля событий с предупреждением",
        "kpi_lead": "Медианный запас времени",
        "kpi_lead_s": "часов до начала события",
        "kpi_prec": "Полезность эпизодов",
        "kpi_prec_s": "эпизодов, за которыми последовало событие",
        "kpi_alerts": "Эпизодов в сутки",
        "kpi_alerts_s": "нагрузка на аналитиков",
        "kpi_net": "Чистый эффект",
        "kpi_net_s": "предотвращённый ущерб минус затраты",
        "kpi_roi": "ROI системы",
        "kpi_roi_s": "чистый эффект / затраты",
        "frozen_delta": "Замороженные пороги дают",
        "status_title": "Текущее состояние каналов",
        "status_state": "Состояние",
        "status_score": "Риск",
        "status_backlog": "Очередь",
        "latest_at": "на",
        "heat_title": "Тепловая карта риска за последние 14 дней",
        "chan_title": "Динамика канала",
        "chan_select": "Компонент",
        "chan_days": "Окно, дней",
        "chan_demand": "Спрос и ёмкость (телеметрия)",
        "chan_backlog": "Очередь и порог перегрузки",
        "chan_score": "Калиброванный риск и пороги",
        "chan_onset": "начало события",
        "models_title": "Сравнение моделей",
        "models_note": "Модель для решений выбрана по чистому эффекту на блоке порогов (не на тесте). Пороги подобраны на том же блоке. Оценка — на тесте, который в подборе не участвовал.",
        "col_model": "Модель", "col_auc": "ROC-AUC", "col_pr": "PR-AUC", "col_bss": "Brier skill",
        "col_recall": "Recall", "col_lead": "Запас, ч", "col_prec": "Полезность", "col_net": "Чистый эффект (тест)",
        "selected": "выбрана",
        "bar_pr": "PR-AUC на тесте", "bar_net": "Чистый эффект на тесте",
        "calib_title": "Калибровка выбранной модели (тест)",
        "calib_note": "Логарифмические оси: вероятности события в каждый час малы.",
        "econ_title": "Экономика решения",
        "econ_water": "От потерь без системы к итогу с системой",
        "econ_break": "Точка безубыточности по полезности эпизода",
        "econ_level": "Уровень", "econ_eps": "Эпизодов", "econ_fact": "Фактическая полезность",
        "econ_be": "Нужная полезность", "econ_sens": "Чистый эффект при разных ущербе и стоимости проверки",
        "econ_ci": "Неопределённость (бутстрап 95%, замороженные параметры)",
        "ci_metric": "Метрика", "ci_value": "Среднее", "ci_range": "95% интервал",
        "dq_title": "Качество данных и дрейф признаков",
        "dq_defects": "Дефекты телеметрии", "dq_rows": "строк с дефектами",
        "psi_title": "Дрейф признаков (PSI, обучение → тест)",
        "psi_note": "PSI < 0.1 — стабильно, 0.1–0.25 — умеренный дрейф, > 0.25 — требуется действие.",
        "scen_title": "Качество предупреждений по сценариям (тест)",
        "scen_col": "Сценарий", "scen_events": "Событий", "scen_caught": "С предупреждением", "scen_recall": "Полнота", "scen_lead": "Запас, ч",
        "imp_title": "Важность признаков градиентного бустинга",
        "method_title": "Методология и ограничения",
        "no_data": "Не найдены файлы результатов в папке ",
        "no_data_hint": "Сначала выполните: python run_capacity_pipeline.py",
        "hours": "ч", "events_unit": "событий", "episodes_unit": "эпизодов",
        "method_md": """
**Что моделируется.** Шесть каналов сети с очередью. Перегрузка — интервал, когда очередь выше порога (220 условных единиц). Событие — начало такого интервала. Стресс-сценарии (постепенное нарастание, каскад, внезапный всплеск, кратковременный всплеск) задают предвестники нагрузки. Перегрузка возникает из динамики очереди, а не задаётся напрямую.

**Что предсказывается.** Начнётся ли событие в ближайшие 6 часов. Часы, когда очередь уже выше порога, из оценки исключены: там предупреждение бессмысленно.

**Защита от утечки.** Признаки используют только прошлые и текущие наблюдения (сдвиги и скользящие окна). Блоки обучения, калибровки, подбора порогов и теста разделены зазором (embargo), метки обрезаны у границ блока (purge).

**Выбор порогов.** Пороги Review и Restrict подбираются на отдельном блоке по чистому эффекту при двух ограничениях: эпизодов в сутки и доля часов в тревоге. Тестовый блок в подборе не участвует.

**Экономика.** Потери без системы — все события по полной стоимости. Предупреждение за 1+ час до события предотвращает долю ущерба, зависящую от уровня реакции. Каждый эпизод стоит фиксированную сумму плюс почасовую стоимость мониторинга. Стоимости и доли предотвращённого ущерба — допущения, а не измерения.

**Ограничения.**
- Данные синтетические, поэтому PSI почти всегда стабилен: в симуляции нет структурных сдвигов.
- Связь «ограничение счёта → перегрузка сети» в модели не моделируется. Это гипотеза, которую проект не проверяет.
- Выборка событий на тесте мала (десятки событий), поэтому интервалы широкие.
- Калибровка в низком диапазоне вероятностей неточная; это видно на графике калибровки.
- Выбор модели по блоку порогов немного оптимистичен для этого блока; тест остаётся чистым.
""",
    },
    "en": {
        "infeasible": 'No admissible thresholds exist for this model: the share of hours in alert mode exceeds the 3% limit. Utility and economic metrics are not reported.', "infeasible_kpi": 'Metrics are not reported: this model has no admissible thresholds.', "no_thr": 'no admissible thresholds', "col_share": 'Alert hours',
        "model_sel": 'Model to view', "model_note": "Switching changes the overview, channel, economics and scenario. The 'selected' mark shows the model chosen by the rule.",
        "ops": 'Operational mode: last hours of the scenario', "ops_now": 'at', "heat_leg": 'Normal — dark · Review — yellow · Restrict — red · Breach (overload) — black · × — event onset', "params": 'Calculation parameters', "open_note": 'Open episodes (outcome not yet known): {n}. Not included in costs.',
        "psi_range": 'PSI range across all features', "feat_title": 'What the features mean', "feat_name": 'Feature', "feat_group": 'Group', "feat_what": 'What it measures', "feat_note": "**PSI** compares a feature's distribution in training and test periods. Near-zero values mean the simulation has no structural shifts. Backlog features are almost always zero, so their quantile bins degenerate and PSI is formally zero. That is a limitation of the method, not evidence of stability.",
        "title": "Payment Network Capacity Risk & Early Warning",
        "subtitle": "Early warning of channel overload in a P2P payment network",
        "synthetic": "All data is synthetic. Results illustrate the methodology and are not evidence of effectiveness on real data.",
        "lang": "Language",
        "thr_box": "Decision thresholds",
        "use_frozen": "Frozen thresholds from the pipeline",
        "thr_review": "Review threshold",
        "thr_restrict": "Restrict threshold",
        "thr_warn": "Restrict threshold is below Review; raised to the Review level.",
        "cost_box": "Economics (conventional units)",
        "missed": "Loss from an event without warning",
        "c_review": "Review cost, per episode",
        "c_restrict": "Restrict cost, per episode",
        "c_hour": "Monitoring cost, per alert hour",
        "s_review": "Share of loss avoided by Review",
        "s_restrict": "Share of loss avoided by Restrict",
        "w_base": "Losses without system", "w_avoid": "Avoided losses", "w_review": "Reviews (Review)", "w_restrict": "Restrictions (Restrict)", "w_total": "Total with system",
        "tabs": ["Overview", "Channel", "Models", "Economics", "Scenario", "Data & drift", "Methodology"],
        "kpi_recall": "Recall",
        "kpi_recall_s": "share of events with a warning",
        "kpi_lead": "Median lead time",
        "kpi_lead_s": "hours before the event",
        "kpi_prec": "Episode precision",
        "kpi_prec_s": "episodes followed by an event",
        "kpi_alerts": "Episodes per day",
        "kpi_alerts_s": "analyst workload",
        "kpi_net": "Net benefit",
        "kpi_net_s": "avoided loss minus costs",
        "kpi_roi": "System ROI",
        "kpi_roi_s": "net benefit / costs",
        "frozen_delta": "Frozen thresholds give",
        "status_title": "Current channel status",
        "status_state": "State",
        "status_score": "Risk",
        "status_backlog": "Backlog",
        "latest_at": "at",
        "heat_title": "Risk heatmap, last 14 days",
        "chan_title": "Channel dynamics",
        "chan_select": "Component",
        "chan_days": "Window, days",
        "chan_demand": "Demand and capacity (telemetry)",
        "chan_backlog": "Backlog and overload threshold",
        "chan_score": "Calibrated risk and thresholds",
        "chan_onset": "event onset",
        "models_title": "Model comparison",
        "models_note": "The decision model is chosen by net benefit on the threshold block, not on the test set. Thresholds were fitted on that same block. Evaluation is on the test set, which was not used for fitting.",
        "col_model": "Model", "col_auc": "ROC-AUC", "col_pr": "PR-AUC", "col_bss": "Brier skill",
        "col_recall": "Recall", "col_lead": "Lead, h", "col_prec": "Precision", "col_net": "Net benefit (test)",
        "selected": "selected",
        "bar_pr": "PR-AUC on test", "bar_net": "Net benefit on test",
        "calib_title": "Calibration of the selected model (test)",
        "calib_note": "Log axes: hourly event probabilities are small.",
        "econ_title": "Decision economics",
        "econ_water": "From losses without the system to the total with the system",
        "econ_break": "Break-even episode precision",
        "econ_level": "Level", "econ_eps": "Episodes", "econ_fact": "Actual precision",
        "econ_be": "Required precision", "econ_sens": "Net benefit under different loss and review costs",
        "econ_ci": "Uncertainty (bootstrap 95%, frozen parameters)",
        "ci_metric": "Metric", "ci_value": "Mean", "ci_range": "95% interval",
        "dq_title": "Data quality and feature drift",
        "dq_defects": "Telemetry defects", "dq_rows": "rows with defects",
        "psi_title": "Feature drift (PSI, training → test)",
        "psi_note": "PSI < 0.1 stable, 0.1–0.25 moderate drift, > 0.25 action needed.",
        "scen_title": "Warning quality by scenario (test)",
        "scen_col": "Scenario", "scen_events": "Events", "scen_caught": "Warned", "scen_recall": "Recall", "scen_lead": "Lead, h",
        "imp_title": "Gradient boosting feature importance",
        "method_title": "Methodology and limitations",
        "no_data": "Result files not found in ",
        "no_data_hint": "First run: python run_capacity_pipeline.py",
        "hours": "h", "events_unit": "events", "episodes_unit": "episodes",
        "method_md": """
**What is modelled.** Six network channels with a queue. Overload is an interval when the queue is above the threshold (220 conventional units). An event is the start of such an interval. Stress scenarios (gradual build-up, cascade, flash surge, transient surge) produce precursors in load. Overload emerges from queue dynamics rather than being set directly.

**What is predicted.** Whether an event starts within the next 6 hours. Hours already in overload are excluded from evaluation, because a warning then has no value.

**Leakage protection.** Features use only past and current observations (lags and trailing windows). Training, calibration, threshold and test blocks are separated by an embargo gap, and labels are truncated at block edges (purge).

**Threshold choice.** Review and Restrict thresholds are fitted on a separate block by net benefit under two limits: episodes per day and share of hours in alert. The test block is not used for fitting.

**Economics.** Losses without the system count every event at full cost. A warning at least one hour ahead avoids a share of the loss that depends on the response level. Each episode costs a fixed amount plus an hourly monitoring cost. Costs and avoided-loss shares are assumptions, not measurements.

**Limitations.**
- The data is synthetic, so PSI is almost always stable: the simulation has no structural shifts.
- The link "account restriction → network overload" is not modelled. It is a hypothesis the project does not test.
- The test set has few events (tens), so intervals are wide.
- Calibration is imprecise at low probabilities; the calibration chart shows this.
- Model selection on the threshold block is slightly optimistic for that block; the test set stays clean.
""",
    },
    "uk": {
        "infeasible": 'Для цієї моделі немає допустимих порогів: частка годин у режимі тривоги перевищує ліміт 3%. Корисність та економіка не розраховуються.', "infeasible_kpi": 'Показники не розраховуються: для цієї моделі немає допустимих порогів.', "no_thr": 'немає допустимих порогів', "col_share": 'Годин у тривозі',
        "model_sel": 'Модель для перегляду', "model_note": 'Перемикання змінює огляд, канал, економіку і сценарій. Позначка «обрано» — модель, обрана за правилом.',
        "ops": 'Оперативний режим: останні години сценарію', "ops_now": 'на', "heat_leg": 'Нормально — темний · Review — жовтий · Restrict — червоний · Breach (перевантаження) — чорний · × — початок події', "params": 'Параметри розрахунку', "open_note": 'Відкритих епізодів (результат ще не настав): {n}. До витрат не включено.',
        "psi_range": 'Діапазон PSI за всіма ознаками', "feat_title": 'Що означають ознаки', "feat_name": 'Ознака', "feat_group": 'Група', "feat_what": 'Що вимірює', "feat_note": '**PSI** порівнює розподіл ознаки на навчальному та тестовому періодах. Майже нульові значення означають, що в симуляції немає структурних зсувів. Ознаки черги (backlog) майже завжди нульові, тому їхні квантильні бини вироджені, і PSI для них формально нульовий. Це обмеження методу, а не ознака стабільності.',
        "title": "Payment Network Capacity Risk & Early Warning",
        "subtitle": "Раннє попередження про перевантаження каналів P2P-платіжної мережі",
        "synthetic": "Усі дані синтетичні. Результати ілюструють методологію і не є доказом ефективності на реальних даних.",
        "lang": "Мова",
        "thr_box": "Пороги рішень",
        "use_frozen": "Заморожені пороги з пайплайна",
        "thr_review": "Поріг Review",
        "thr_restrict": "Поріг Restrict",
        "thr_warn": "Поріг Restrict нижчий за поріг Review; піднято до рівня Review.",
        "cost_box": "Економіка (умовні одиниці)",
        "missed": "Збиток від події без попередження",
        "c_review": "Перевірка (Review), на епізод",
        "c_restrict": "Обмеження (Restrict), на епізод",
        "c_hour": "Моніторинг, за годину тривоги",
        "s_review": "Частка збитку, яку запобігає Review",
        "s_restrict": "Частка збитку, яку запобігає Restrict",
        "w_base": "Збитки без системи", "w_avoid": "Запобігнуті збитки", "w_review": "Перевірки (Review)", "w_restrict": "Обмеження (Restrict)", "w_total": "Підсумок з системою",
        "tabs": ["Огляд", "Канал", "Моделі", "Економіка", "Сценарій", "Дані та дрейф", "Методологія"],
        "kpi_recall": "Повнота (recall)",
        "kpi_recall_s": "частка подій із попередженням",
        "kpi_lead": "Медіанний запас часу",
        "kpi_lead_s": "годин до початку події",
        "kpi_prec": "Корисність епізодів",
        "kpi_prec_s": "епізодів, після яких настала подія",
        "kpi_alerts": "Епізодів на добу",
        "kpi_alerts_s": "навантаження на аналітиків",
        "kpi_net": "Чистий ефект",
        "kpi_net_s": "запобігнутий збиток мінус витрати",
        "kpi_roi": "ROI системи",
        "kpi_roi_s": "чистий ефект / витрати",
        "frozen_delta": "Заморожені пороги дають",
        "status_title": "Поточний стан каналів",
        "status_state": "Стан",
        "status_score": "Ризик",
        "status_backlog": "Черга",
        "latest_at": "на",
        "heat_title": "Теплова карта ризику за 14 днів",
        "chan_title": "Динаміка каналу",
        "chan_select": "Компонент",
        "chan_days": "Вікно, днів",
        "chan_demand": "Попит і потужність (телеметрія)",
        "chan_backlog": "Черга і поріг перевантаження",
        "chan_score": "Калібрований ризик і пороги",
        "chan_onset": "початок події",
        "models_title": "Порівняння моделей",
        "models_note": "Модель для рішень обрано за чистим ефектом на блоці порогів, а не на тесті. Пороги підібрано на тому самому блоці. Оцінка — на тесті, який у підборі не брав участі.",
        "col_model": "Модель", "col_auc": "ROC-AUC", "col_pr": "PR-AUC", "col_bss": "Brier skill",
        "col_recall": "Recall", "col_lead": "Запас, год", "col_prec": "Корисність", "col_net": "Чистий ефект (тест)",
        "selected": "обрано",
        "bar_pr": "PR-AUC на тесті", "bar_net": "Чистий ефект на тесті",
        "calib_title": "Калібрування обраної моделі (тест)",
        "calib_note": "Логарифмічні осі: погодинні ймовірності подій малі.",
        "econ_title": "Економіка рішення",
        "econ_water": "Від збитків без системи до підсумку з системою",
        "econ_break": "Точка беззбитковості корисності епізоду",
        "econ_level": "Рівень", "econ_eps": "Епізодів", "econ_fact": "Фактична корисність",
        "econ_be": "Потрібна корисність", "econ_sens": "Чистий ефект за різного збитку та вартості перевірки",
        "econ_ci": "Невизначеність (бутстрап 95%, заморожені параметри)",
        "ci_metric": "Метрика", "ci_value": "Середнє", "ci_range": "95% інтервал",
        "dq_title": "Якість даних і дрейф ознак",
        "dq_defects": "Дефекти телеметрії", "dq_rows": "рядків із дефектами",
        "psi_title": "Дрейф ознак (PSI, навчання → тест)",
        "psi_note": "PSI < 0.1 стабільно, 0.1–0.25 помірний дрейф, > 0.25 потрібна дія.",
        "scen_title": "Якість попереджень за сценаріями (тест)",
        "scen_col": "Сценарій", "scen_events": "Подій", "scen_caught": "З попередженням", "scen_recall": "Повнота", "scen_lead": "Запас, год",
        "imp_title": "Важливість ознак градієнтного бустингу",
        "method_title": "Методологія та обмеження",
        "no_data": "Файли результатів не знайдено в ",
        "no_data_hint": "Спершу виконайте: python run_capacity_pipeline.py",
        "hours": "год", "events_unit": "подій", "episodes_unit": "епізодів",
        "method_md": """
**Що моделюється.** Шість каналів мережі з чергою. Перевантаження — інтервал, коли черга вища за поріг (220 умовних одиниць). Подія — початок такого інтервалу. Сценарії стресу (поступове наростання, каскад, раптовий сплеск, короткочасний сплеск) дають передвісники навантаження. Перевантаження виникає з динаміки черги, а не задається напряму.

**Що прогнозується.** Чи почнеться подія протягом наступних 6 годин. Години, коли черга вже вища за поріг, з оцінки виключено: попередження там не має сенсу.

**Захист від витоку.** Ознаки використовують лише минулі та поточні спостереження (лаги і ковзні вікна). Блоки навчання, калібрування, підбору порогів і тесту розділені зазором (embargo), мітки обрізано на межах блоків (purge).

**Вибір порогів.** Пороги Review і Restrict підбираються на окремому блоці за чистим ефектом із двома обмеженнями: епізодів на добу і частка годин у тривозі. Тестовий блок у підборі не бере участі.

**Економіка.** Збитки без системи враховують кожну подію за повною вартістю. Попередження щонайменше за годину до події запобігає частці збитку, що залежить від рівня реакції. Кожен епізод коштує фіксовану суму плюс погодинні витрати на моніторинг. Вартості та частки запобіганого збитку — припущення, а не вимірювання.

**Обмеження.**
- Дані синтетичні, тому PSI майже завжди стабільний: у симуляції немає структурних зсувів.
- Зв'язок «обмеження рахунку → перевантаження мережі» в моделі не відтворено. Це гіпотеза, яку проєкт не перевіряє.
- Вибірка подій на тесті мала (десятки подій), тому інтервали широкі.
- Калібрування в низькому діапазоні ймовірностей неточне; це видно на графіку калібрування.
- Вибір моделі за блоком порогів трохи оптимістичний для цього блоку; тест лишається чистим.
""",
    },
}

LANG_OPTS = {"English": "en", "Українська": "uk", "Русский": "ru"}


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------
REQUIRED = [
    "frozen_metrics.json", "test_predictions.csv", "event_catalog.csv", "test_events.csv",
    "test_episodes.csv", "model_comparison.csv", "threshold_grid.csv", "feature_psi_report.csv",
    "feature_importance.csv", "calibration_points.csv", "scenario_breakdown.csv", "data_summary.json",
]


def _load_scen_hourly(d: Path):
    p = d / "scenario_hourly.csv"
    if not p.exists():
        return None
    sh = pd.read_csv(p)
    sh["timestamp_utc"] = pd.to_datetime(sh["timestamp_utc"], utc=True)
    return sh


@st.cache_data(show_spinner=False)
def load_results(dir_str: str):
    d = Path(dir_str)
    missing = [f for f in REQUIRED if not (d / f).exists()]
    if missing:
        return None
    frozen = json.loads((d / "frozen_metrics.json").read_text(encoding="utf-8"))
    tp = pd.read_csv(d / "test_predictions.csv")
    tp["timestamp_utc"] = pd.to_datetime(tp["timestamp_utc"], utc=True)
    cat = pd.read_csv(d / "event_catalog.csv")
    cat["onset_ts"] = pd.to_datetime(cat["onset_ts"], utc=True)
    return {
        "frozen": frozen,
        "cfg": Config(**frozen["config"]),
        "tp": tp,
        "cat": cat,
        "test_events": pd.read_csv(d / "test_events.csv"),
        "test_episodes": pd.read_csv(d / "test_episodes.csv"),
        "cmp": pd.read_csv(d / "model_comparison.csv"),
        "grid": pd.read_csv(d / "threshold_grid.csv"),
        "psi": pd.read_csv(d / "feature_psi_report.csv"),
        "imp": pd.read_csv(d / "feature_importance.csv"),
        "calib": pd.read_csv(d / "calibration_points.csv"),
        "scen": pd.read_csv(d / "scenario_breakdown.csv"),
        "summary": json.loads((d / "data_summary.json").read_text(encoding="utf-8")),
        "scen_meta": json.loads((d / "scenario_meta.json").read_text(encoding="utf-8"))
        if (d / "scenario_meta.json").exists() else None,
        "scen_hourly": _load_scen_hourly(d),
        "scen_events": pd.read_csv(d / "scenario_events.csv") if (d / "scenario_events.csv").exists() else None,
    }


MODEL_NAMES = {
    "en": {"no_skill_constant": "No-skill constant", "rule_backlog_level": "Queue rule",
           "logistic_regression": "Logistic regression", "gradient_boosting": "Gradient boosting (HistGB)"},
    "ru": {"no_skill_constant": "Без навыка (константа)", "rule_backlog_level": "Правило по очереди",
           "logistic_regression": "Логистическая регрессия", "gradient_boosting": "Градиентный бустинг (HistGB)"},
    "uk": {"no_skill_constant": "Без навички (константа)", "rule_backlog_level": "Правило за чергою",
           "logistic_regression": "Логістична регресія", "gradient_boosting": "Градієнтний бустинг (HistGB)"},
}

FEAT_TXT = {
    "ru": {"base": {"demand": "спрос", "capacity": "ёмкость", "backlog": "очередь", "util": "загрузка (спрос / ёмкость)"},
           "suf": {"now": "текущее значение", "lag1": "значение час назад", "d3h": "изменение за последние 3 часа",
                   "roll6": "среднее за 6 часов", "roll12": "среднее за 12 часов", "roll24": "среднее за 24 часа"},
           "special": {"backlog_slope6": "изменение очереди за 6 часов", "backlog_slope12": "изменение очереди за 12 часов",
                       "clearance_h": "часов до очистки очереди при текущей ёмкости", "dq_flag": "флаг дефекта телеметрии",
                       "hour_of_day": "час суток (UTC)", "day_of_week": "день недели (0 = понедельник)"},
           "group": "Календарь", "groups": {"cal": "Календарь", "dq": "Качество данных", "trend": "Тренд", "cur": "Текущее", "past": "Прошлое", "roll": "Скользящее среднее"}},
    "en": {"base": {"demand": "demand", "capacity": "capacity", "backlog": "backlog", "util": "utilisation (demand / capacity)"},
           "suf": {"now": "current value", "lag1": "value one hour ago", "d3h": "change over the last 3 hours",
                   "roll6": "mean over 6 hours", "roll12": "mean over 12 hours", "roll24": "mean over 24 hours"},
           "special": {"backlog_slope6": "backlog change over 6 hours", "backlog_slope12": "backlog change over 12 hours",
                       "clearance_h": "hours to clear the backlog at current capacity", "dq_flag": "telemetry defect flag",
                       "hour_of_day": "hour of day (UTC)", "day_of_week": "day of week (0 = Monday)"},
           "group": "Calendar", "groups": {"cal": "Calendar", "dq": "Data quality", "trend": "Trend", "cur": "Current", "past": "Past", "roll": "Rolling mean"}},
    "uk": {"base": {"demand": "попит", "capacity": "потужність", "backlog": "черга", "util": "завантаження (попит / потужність)"},
           "suf": {"now": "поточне значення", "lag1": "значення годину тому", "d3h": "зміна за останні 3 години",
                   "roll6": "середнє за 6 годин", "roll12": "середнє за 12 годин", "roll24": "середнє за 24 години"},
           "special": {"backlog_slope6": "зміна черги за 6 годин", "backlog_slope12": "зміна черги за 12 годин",
                       "clearance_h": "годин до очищення черги за поточної потужності", "dq_flag": "прапорець дефекту телеметрії",
                       "hour_of_day": "година доби (UTC)", "day_of_week": "день тижня (0 = понеділок)"},
           "group": "Календар", "groups": {"cal": "Календар", "dq": "Якість даних", "trend": "Тренд", "cur": "Поточне", "past": "Минуле", "roll": "Ковзне середнє"}},
}


def feat_description(name: str, lang: str) -> tuple[str, str]:
    """Описание признака и его группа по имени колонки."""
    t = FEAT_TXT[lang]
    g = t["groups"]
    if name in t["special"]:
        grp = g["cal"] if name in ("hour_of_day", "day_of_week") else (
            g["dq"] if name == "dq_flag" else g["trend"])
        return t["special"][name], grp
    base, _, suf = name.partition("_")
    if base in t["base"] and suf in t["suf"]:
        grp = {"now": g["cur"], "lag1": g["past"], "d3h": g["trend"]}.get(suf, g["roll"])
        return f'{t["base"][base]}: {t["suf"][suf]}', grp
    return name, "—"


def fmt_money(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def kpi(label: str, value: str, sub: str) -> None:
    st.markdown(
        f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
        f'<div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def pill(state: str) -> str:
    return f'<span class="pill" style="background:{COL.get(state, "#64748b")}">{state}</span>'


CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1400px;}
.hero {background: linear-gradient(135deg, #1e3a8a 0%, #0f766e 100%); color: #fff;
       border-radius: 16px; padding: 18px 24px; margin-bottom: 12px;}
.hero h2 {color: #fff; margin: 0 0 4px 0; font-size: 1.55rem;}
.hero p {margin: 0; opacity: .92;}
.kpi {background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 14px 16px;
      box-shadow: 0 1px 2px rgba(15,23,42,.05); height: 100%;}
.kpi .l {font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: .05em;}
.kpi .v {font-size: 26px; font-weight: 700; color: #0f172a; margin-top: 2px;}
.kpi .s {font-size: 12px; color: #64748b;}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px;
       font-weight: 600; color: #fff;}
.card {background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 12px 14px;
       box-shadow: 0 1px 2px rgba(15,23,42,.05);}
.card .cid {font-weight: 700; color: #0f172a;}
.card .meta {font-size: 12px; color: #64748b; margin-top: 4px;}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Основной поток
# ---------------------------------------------------------------------------
lang_label = st.sidebar.radio("Language / Мова / Язык", list(LANG_OPTS), horizontal=True)
L = LANG_OPTS[lang_label]
TX = T[L]

data = load_results(str(DATA_DIR))
if data is None:
    st.error(TX["no_data"] + str(DATA_DIR))
    st.info(TX["no_data_hint"])
    st.stop()

frozen = data["frozen"]
cfg0: Config = data["cfg"]
tp = data["tp"]
cat = data["cat"]
chosen = frozen["chosen_model"]
cmp_all = data["cmp"].set_index("model")
selectable = [m for m in ("logistic_regression", "gradient_boosting", "rule_backlog_level")
              if m in cmp_all.index and pd.notna(cmp_all.loc[m, "t_review"])]
test_block = tuple(frozen["policy_block"])
onsets = cat["onset_hour"].to_numpy()

st.sidebar.markdown("---")
st.sidebar.subheader(TX["thr_box"])
use_frozen = st.sidebar.toggle(TX["use_frozen"], value=True)
sel_model = st.sidebar.selectbox(
    TX["model_sel"], selectable, index=selectable.index(chosen),
    format_func=lambda m: MODEL_NAMES[L][m])
score_col = f"score_{sel_model}"
fz_rev = float(cmp_all.loc[sel_model, "t_review"])
fz_res = float(cmp_all.loc[sel_model, "t_restrict"])
sel_feasible = bool(cmp_all.loc[sel_model, "feasible_under_alert_limit"])
if use_frozen:
    t_rev, t_res = fz_rev, fz_res
else:
    t_rev = st.sidebar.slider(TX["thr_review"], 0.0, 1.0, fz_rev, 0.0005, format="%.4f")
    t_res = st.sidebar.slider(TX["thr_restrict"], 0.0, 1.0, fz_res, 0.0005, format="%.4f")
if t_res < t_rev:
    st.sidebar.warning(TX["thr_warn"])
    t_res = t_rev

st.sidebar.subheader(TX["cost_box"])
missed = st.sidebar.number_input(TX["missed"], 10_000, 500_000, int(cfg0.missed_event_loss), 5_000)
c_rev = st.sidebar.number_input(TX["c_review"], 500, 20_000, int(cfg0.review_cost), 500)
c_res = st.sidebar.number_input(TX["c_restrict"], 1_000, 50_000, int(cfg0.restrict_action_cost), 500)
c_hour = st.sidebar.number_input(TX["c_hour"], 0, 1_000, int(cfg0.alert_hour_cost), 10)
s_rev = st.sidebar.slider(TX["s_review"], 0.0, 1.0, float(cfg0.success_review), 0.05)
s_res = st.sidebar.slider(TX["s_restrict"], 0.0, 1.0, float(cfg0.success_restrict), 0.05)

cfg = replace(cfg0, missed_event_loss=float(missed), review_cost=float(c_rev),
              restrict_action_cost=float(c_res), alert_hour_cost=float(c_hour),
              success_review=float(s_rev), success_restrict=float(s_res))

# что-если: пересчёт политики на тесте теми же функциями, что и в пайплайне
ok_mask = tp["eval_ok"].astype(bool)
tp = tp.assign(live_state=np.where(~ok_mask, "Breach",
                                   np.where(tp[score_col] >= t_res, "Restrict",
                                            np.where(tp[score_col] >= t_rev, "Review", "Normal"))))
te = tp[ok_mask]
m, ev_df, ep_df = core.policy_outcome(
    te["hour"].to_numpy(), te[score_col].to_numpy(), onsets, cfg, t_rev, t_res, test_block)
econ = core.reprice(ev_df, ep_df, cfg)
fm = frozen["event_metrics_test"]

st.markdown(
    f'<div class="hero"><h2>{TX["title"]}</h2><p>{TX["subtitle"]}</p></div>',
    unsafe_allow_html=True,
)
st.info(TX["synthetic"])
if not sel_feasible:
    st.warning(TX["infeasible"])

# оперативный поток: сценарий (последние часы — «сейчас»); без сценария — тестовый блок
if data["scen_hourly"] is not None and data["scen_meta"] is not None and data["scen_events"] is not None:
    ops = data["scen_hourly"].rename(columns={"demand": "demand_obs", "capacity": "capacity_obs"})
    ops = ops.assign(score=ops[score_col])
    ops["live_state"] = np.where(ops["backlog_truth"] > cfg.backlog_threshold, "Breach",
                                 np.where(ops["score"] >= t_res, "Restrict",
                                          np.where(ops["score"] >= t_rev, "Review", "Normal")))
    ts0_ops = pd.Timestamp(cfg.start_ts, tz="UTC")
    ops_events = data["scen_events"].copy()
    blk = data["scen_meta"]["block"]
    ops_events = ops_events[(ops_events["onset_hour"] >= blk[0]) & (ops_events["onset_hour"] < blk[1])]
    ops_events = ops_events.assign(onset_ts=[ts0_ops + pd.Timedelta(hours=int(h)) for h in ops_events["onset_hour"]])
    ops_mode = True
else:
    ops = tp.assign(score=tp[score_col])
    ops_events = cat.assign(onset_ts=cat["onset_ts"])
    ops_mode = False
ops = ops.sort_values(["component_id", "timestamp_utc"])
ops_events_cols = ops_events[["component_id", "onset_ts"]]

tabs = st.tabs(TX["tabs"])

# ---------------------------------------------------------------------------
# 1. Обзор
# ---------------------------------------------------------------------------
with tabs[0]:
    if not sel_feasible:
        st.info(TX["infeasible_kpi"])
    else:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            kpi(TX["kpi_recall"], f'{m["recall"]:.0%}' if m["n_events"] else "—",
                f'{m["n_caught"]} / {m["n_events"]} {TX["events_unit"]}')
        with c2:
            kpi(TX["kpi_lead"], f'{m["median_lead_h"]:.1f} {TX["hours"]}' if m["n_caught"] else "—", TX["kpi_lead_s"])
        with c3:
            kpi(TX["kpi_prec"], f'{m["precision"]:.0%}' if m["n_episodes"] else "—", TX["kpi_prec_s"])
        with c4:
            kpi(TX["kpi_alerts"], f'{m["alerts_per_day"]:.2f}', TX["kpi_alerts_s"])
        with c5:
            kpi(TX["kpi_net"], fmt_money(econ["net_benefit"]), TX["kpi_net_s"])
        with c6:
            roi = econ["roi"]
            kpi(TX["kpi_roi"], f"{roi:.1f}×" if roi == roi else "—", TX["kpi_roi_s"])
        if sel_model == chosen and (t_rev, t_res) != (fz_rev, fz_res):
            st.caption(f'{TX["frozen_delta"]}: recall {fm["recall"]:.0%}, '
                       f'net {fmt_money(frozen["economics_test"]["net_benefit"])}.')

    st.markdown(f'#### {TX["status_title"]}')
    if ops_mode:
        st.caption(f'{TX["ops"]}')
    last = ops.groupby("component_id").tail(1)
    cards = st.columns(len(last))
    for col, (_, row) in zip(cards, last.iterrows()):
        with col:
            st.markdown(
                f'<div class="card"><div class="cid">{row["component_id"]}</div>'
                f'<div style="margin-top:6px">{pill(row["live_state"])}</div>'
                f'<div class="meta">{TX["status_score"]}: <b>{row["score"]:.3f}</b></div>'
                f'<div class="meta">{TX["status_backlog"]}: <b>{row["backlog_truth"]:.0f}</b></div>'
                f'<div class="meta">{TX["latest_at"]} {row["timestamp_utc"]:%d.%m.%Y %H:%M} UTC</div></div>',
                unsafe_allow_html=True)

    st.markdown(f'#### {TX["heat_title"]}')
    end = ops["timestamp_utc"].max()
    recent = ops[ops["timestamp_utc"] > end - pd.Timedelta(days=14)]
    code = {"Normal": 0, "Review": 1, "Restrict": 2, "Breach": 3}
    recent = recent.assign(code=recent["live_state"].map(code))
    piv = recent.pivot_table(index="component_id", columns="timestamp_utc", values="code")
    spiv = recent.pivot_table(index="component_id", columns="timestamp_utc", values="score")
    fig_h = go.Figure(go.Heatmap(
        z=piv.to_numpy(), x=piv.columns, y=piv.index, zmin=0, zmax=3,
        colorscale=[[0, "#1e3a5f"], [0.249, "#1e3a5f"], [0.25, "#f59e0b"], [0.499, "#f59e0b"],
                    [0.5, "#ef4444"], [0.749, "#ef4444"], [0.75, "#0b0b0b"], [1, "#0b0b0b"]],
        showscale=False, customdata=spiv.to_numpy(),
        hovertemplate="%{y}<br>%{x|%d.%m %H:%M} UTC<br>risk %{customdata:.3f}<extra></extra>",
        xgap=0, ygap=2))
    ev_r = ops_events_cols[ops_events_cols["onset_ts"] > end - pd.Timedelta(days=14)]
    if len(ev_r):
        fig_h.add_trace(go.Scatter(
            x=ev_r["onset_ts"], y=ev_r["component_id"], mode="markers", name=TX["chan_onset"],
            marker=dict(symbol="x", size=13, color="#ffffff", line=dict(color="#0b0b0b", width=1.5)),
            hovertemplate="%{y} — " + TX["chan_onset"] + "<br>%{x|%d.%m %H:%M} UTC<extra></extra>"))
    fig_h.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
                        legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_h, width="stretch")
    st.caption(TX["heat_leg"] if ops_mode else "")

# ---------------------------------------------------------------------------
# 2. Канал
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown(f'#### {TX["chan_title"]}')
    if ops_mode:
        st.caption(TX["ops"])
    cc1, cc2 = st.columns([1, 2])
    comps = sorted(ops["component_id"].unique())
    default_comp = "COMP-06" if "COMP-06" in comps else comps[0]
    comp = cc1.selectbox(TX["chan_select"], comps, index=comps.index(default_comp))
    days = cc2.slider(TX["chan_days"], 1, 18, 7)
    ch = ops[ops["component_id"] == comp].sort_values("timestamp_utc")
    end = ch["timestamp_utc"].max()
    ch = ch[ch["timestamp_utc"] > end - pd.Timedelta(days=days)]
    ch_ev = ops_events_cols[(ops_events_cols["component_id"] == comp)
                            & (ops_events_cols["onset_ts"] > ch["timestamp_utc"].min())]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.32, 0.30, 0.38],
                        subplot_titles=(TX["chan_demand"], TX["chan_backlog"], TX["chan_score"]))
    fig.add_trace(go.Scatter(x=ch["timestamp_utc"], y=ch["demand_obs"], name="demand",
                             line=dict(color=NAVY, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ch["timestamp_utc"], y=ch["capacity_obs"], name="capacity",
                             line=dict(color=TEAL, width=1.6, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ch["timestamp_utc"], y=ch["backlog_truth"], name="backlog",
                             line=dict(color="#475569", width=1.6)), row=2, col=1)
    fig.add_hline(y=cfg.backlog_threshold, line_color=COL["Restrict"], line_dash="dash", row=2, col=1)
    fig.add_trace(go.Scatter(x=ch["timestamp_utc"], y=ch["score"], name="risk",
                             line=dict(color=NAVY, width=1.6)), row=3, col=1)
    fig.add_hline(y=t_rev, line_color=COL["Review"], line_dash="dash", row=3, col=1)
    fig.add_hline(y=t_res, line_color=COL["Restrict"], line_dash="dash", row=3, col=1)
    for st_name in ("Review", "Restrict"):
        sel = ch[ch["live_state"] == st_name]
        fig.add_trace(go.Scatter(x=sel["timestamp_utc"], y=sel["score"], mode="markers", name=st_name,
                                 marker=dict(color=COL[st_name], size=7)), row=3, col=1)
    for _, ev in ch_ev.iterrows():
        fig.add_vline(x=ev["onset_ts"], line_color="#64748b", line_dash="dot", row="all", col=1)
    fig.update_layout(template="plotly_white", height=640, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.08))
    fig.update_yaxes(type="log", row=3, col=1)
    st.plotly_chart(fig, width="stretch")
    st.caption(f'{TX["chan_onset"]}: {len(ch_ev)}')

# ---------------------------------------------------------------------------
# 3. Модели
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown(f'#### {TX["models_title"]}')
    st.info(TX["models_note"])
    st.caption(TX["model_note"])
    cmp_df = data["cmp"].copy()
    cmp_df["name"] = cmp_df["model"].map(MODEL_NAMES[L]).fillna(cmp_df["model"])
    infeas = ~cmp_df["feasible_under_alert_limit"].astype(bool)
    show = pd.DataFrame({
        TX["col_model"]: cmp_df["name"],
        TX["col_auc"]: cmp_df["roc_auc_test"].round(3),
        TX["col_pr"]: cmp_df["pr_auc_test"].round(3),
        TX["col_bss"]: cmp_df["brier_skill_test"].round(3),
        TX["col_recall"]: cmp_df["recall_test"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—"),
        TX["col_lead"]: cmp_df["median_lead_h_test"].round(1),
        TX["col_prec"]: cmp_df["precision_test"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—"),
        TX["col_net"]: cmp_df["net_benefit_test"].map(lambda v: fmt_money(v) if pd.notna(v) else "—"),
        TX["selected"]: cmp_df["selected"].map({True: "✔", False: ""}),
    })
    show.insert(len(show.columns) - 1, TX["col_share"],
                cmp_df["alert_hour_share_test"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—"))
    def _pct(v):
        return f"{v:.0%}" if pd.notna(v) else "—"
    fmt_cols = {
        TX["col_recall"]: cmp_df["recall_test"].map(_pct),
        TX["col_lead"]: cmp_df["median_lead_h_test"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—"),
        TX["col_prec"]: cmp_df["precision_test"].map(_pct),
        TX["col_net"]: cmp_df["net_benefit_test"].map(lambda v: fmt_money(v) if pd.notna(v) else "—"),
    }
    for col, ser in fmt_cols.items():
        show[col] = ser.astype(str).to_numpy()
        show.loc[infeas.to_numpy(), col] = TX["no_thr"]
    st.dataframe(show, hide_index=True, width="stretch")

    b1, b2 = st.columns(2)
    names = cmp_df["name"]
    with b1:
        f1 = go.Figure(go.Bar(x=names, y=cmp_df["pr_auc_test"], marker_color=NAVY))
        f1.update_layout(template="plotly_white", title=TX["bar_pr"], height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f1, width="stretch")
    with b2:
        f2 = go.Figure(go.Bar(x=names, y=cmp_df["net_benefit_test"], marker_color=TEAL))
        f2.update_layout(template="plotly_white", title=TX["bar_net"], height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f2, width="stretch")

    st.markdown(f'#### {TX["calib_title"]}')
    st.caption(TX["calib_note"])
    cal = data["calib"]
    fc = go.Figure()
    lo_v = max(float(cal["predicted"].min()) * 0.5, 1e-5)
    hi_v = float(max(cal["predicted"].max(), cal["observed"].max())) * 1.5
    fc.add_trace(go.Scatter(x=[lo_v, hi_v], y=[lo_v, hi_v], mode="lines", name="ideal",
                            line=dict(color="#94a3b8", dash="dash")))
    fc.add_trace(go.Scatter(x=cal["predicted"], y=cal["observed"], mode="markers+lines", name=chosen,
                            marker=dict(size=np.clip(cal["n"] / 40, 6, 18), color=NAVY),
                            text=cal["n"], hovertemplate="pred %{x:.4f}<br>obs %{y:.4f}<br>n=%{text}<extra></extra>"))
    fc.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=10, b=10),
                     xaxis=dict(type="log", title="predicted"), yaxis=dict(type="log", title="observed"))
    st.plotly_chart(fc, width="stretch")

# ---------------------------------------------------------------------------
# 4. Экономика
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown(f'#### {TX["econ_water"]}')
    steps = core.economics_steps(econ)
    step_names = {"baseline": TX["w_base"], "avoided": TX["w_avoid"], "review": TX["w_review"],
                  "restrict": TX["w_restrict"], "total": TX["w_total"]}
    fw = go.Figure(go.Waterfall(
        x=[step_names[s] for s in steps["step"]], y=steps["value"], measure=steps["kind"],
        text=[fmt_money(v) for v in steps["value"]], textposition="outside",
        connector=dict(line=dict(color="#94a3b8")),
        increasing=dict(marker=dict(color="#ef4444")), decreasing=dict(marker=dict(color="#22c55e")),
        totals=dict(marker=dict(color=NAVY))))
    fw.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fw, width="stretch")

    prec = core.precision_by_level(ep_df, cfg)
    st.markdown(f'#### {TX["econ_break"]}')
    rows = []
    for key, lvl_name in (("review", "Review"), ("restrict", "Restrict")):
        p_ = prec[key]
        rows.append({
            TX["econ_level"]: lvl_name, TX["econ_eps"]: p_["episodes"],
            TX["econ_fact"]: f'{p_["precision"]:.0%}' if p_["episodes"] else "—",
            TX["econ_be"]: f'{p_["breakeven_precision"]:.1%}',
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown(f'#### {TX["econ_sens"]}')
    missed_vals = [missed * f for f in (0.5, 0.75, 1.0, 1.5, 2.0)]
    review_vals = [c_rev * f for f in (0.4, 1.0, 2.0, 4.0)]
    grid_vals = np.zeros((len(missed_vals), len(review_vals)))
    for i, mv in enumerate(missed_vals):
        for j, rv in enumerate(review_vals):
            cfg_ij = replace(cfg, missed_event_loss=mv, review_cost=rv)
            grid_vals[i, j] = core.reprice(ev_df, ep_df, cfg_ij)["net_benefit"]
    fs_ = go.Figure(go.Heatmap(
        z=grid_vals, x=[fmt_money(v) for v in review_vals], y=[fmt_money(v) for v in missed_vals],
        colorscale=[[0, "#fca5a5"], [0.5, "#f8fafc"], [1, "#86efac"]], zmid=0,
        text=[[fmt_money(v) for v in row] for row in grid_vals], texttemplate="%{text}",
        colorbar=dict(title="net", thickness=12)))
    fs_.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=TX["c_review"], yaxis_title=TX["missed"])
    st.plotly_chart(fs_, width="stretch")

    st.markdown(f'#### {TX["econ_ci"]}')
    ci = frozen["ci95_test"]
    labels = {"recall": TX["col_recall"], "median_lead_h": TX["col_lead"], "precision": TX["col_prec"],
              "net_benefit": TX["col_net"], "roi": "ROI"}
    ci_rows = []
    for k, lab in labels.items():
        v = ci[k]
        if k in ("recall", "precision"):
            fmt = lambda x: f"{x:.0%}"
        elif k == "median_lead_h":
            fmt = lambda x: f"{x:.1f}"
        elif k == "roi":
            fmt = lambda x: f"{x:.2f}×"
        else:
            fmt = fmt_money
        ci_rows.append({TX["ci_metric"]: lab, TX["ci_value"]: fmt(v["mean"]),
                        TX["ci_range"]: f'{fmt(v["low"])} — {fmt(v["high"])}'})
    st.dataframe(pd.DataFrame(ci_rows), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# 5. Сценарий решений
# ---------------------------------------------------------------------------
with tabs[4]:
    S = SCN[L]
    st.markdown(f'#### {S["head"]}')
    st.info(S["note"])
    if data["scen_hourly"] is None or data["scen_meta"] is None or data["scen_events"] is None:
        st.warning(S["missing"])
    else:
        sh = data["scen_hourly"].copy()
        sh["score"] = sh[score_col]
        sh["live_state"] = np.where(sh["backlog_truth"] > cfg.backlog_threshold, "Breach",
                                    np.where(sh["score"] >= t_res, "Restrict",
                                             np.where(sh["score"] >= t_rev, "Review", "Normal")))
        block = tuple(data["scen_meta"]["block"])
        econ_s, ep_s = core.scenario_policies(sh, data["scen_events"], cfg, t_rev, t_res, block)
        names = {"no_system": S["pol_no"], "review_only": S["pol_review"], "two_tier": S["pol_two"]}

        gain_rows = econ_s.set_index("policy")
        net_two = float(gain_rows.loc["two_tier", "net_benefit"])
        net_rev = float(gain_rows.loc["review_only", "net_benefit"])
        st.markdown(
            f'<div class="hero"><div style="font-size:12px;opacity:.85;text-transform:uppercase;letter-spacing:.06em">'
            f'{S["gain"]}</div><div style="font-size:34px;font-weight:800;margin-top:2px">{fmt_money(net_two)}</div>'
            f'<div style="opacity:.92">{S["gain_s"]}: +{fmt_money(net_two - net_rev)}</div></div>',
            unsafe_allow_html=True)
        st.markdown(f'#### {S["pols"]}')
        head = [S["col_policy"], S["col_events"], S["col_caught"], S["col_avoided"],
                S["col_costs"], S["col_net"], S["col_roi"]]
        body = ""
        for _, r in econ_s.iterrows():
            hl = r["policy"] == "two_tier"
            cells = [names[r["policy"]], str(int(r["events"])), str(int(r["caught"])),
                     fmt_money(r["avoided_loss"]), fmt_money(r["alert_cost"]), fmt_money(r["net_benefit"]),
                     f'{r["roi"]:.2f}×' if pd.notna(r["roi"]) else "—"]
            tds = "".join(f"<td>{c}</td>" for c in cells)
            body += f'<tr class="{"hl" if hl else ""}">{tds}</tr>'
        ths = "".join(f"<th>{h}</th>" for h in head)
        css = ("<style>.ptbl{width:100%;border-collapse:separate;border-spacing:0;font-size:15px;"
               "border:1px solid #e2e8f0;border-radius:12px;overflow:hidden}"
               ".ptbl th{background:#1e293b;color:#fff;padding:10px 12px;text-align:right;font-weight:600}"
               ".ptbl th:first-child,.ptbl td:first-child{text-align:left}"
               ".ptbl td{padding:11px 12px;text-align:right;border-top:1px solid #e2e8f0;color:#0f172a}"
               ".ptbl tr.hl td{background:#dcfce7;font-weight:700;color:#14532d}</style>")
        html = f'{css}<table class="ptbl"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
        st.markdown(html, unsafe_allow_html=True)
        st.caption(f'{TX["params"]}: ' + f'{fmt_money(cfg.missed_event_loss)} · '
                   f'{fmt_money(cfg.review_cost)} / {fmt_money(cfg.restrict_action_cost)} · '
                   f'{cfg.success_review:.0%} / {cfg.success_restrict:.0%}. '
                   + TX["open_note"].format(n=int(econ_s["open_episodes"].max())))
        fb = go.Figure(go.Bar(x=econ_s["policy"].map(names), y=econ_s["net_benefit"], marker_color=TEAL))
        fb.update_layout(template="plotly_white", height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fb, width="stretch")

        ts0_ = pd.Timestamp(cfg.start_ts, tz="UTC")
        focus = list(data["scen_meta"]["schedule"].keys())
        for comp in focus:
            st.markdown(f'#### {S["focus"]}: {comp}')
            g = sh[sh["component_id"] == comp].sort_values("hour")
            evs = data["scen_events"]
            evs = evs[(evs["component_id"] == comp) & (evs["onset_hour"] >= block[0]) & (evs["onset_hour"] < block[1])]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                                row_heights=[0.55, 0.45], subplot_titles=(S["chart_risk"], S["chart_bl"]))
            fig.add_trace(go.Scatter(x=g["timestamp_utc"], y=g["score"], name="risk",
                                     line=dict(color=NAVY, width=1.6)), row=1, col=1)
            fig.add_hline(y=t_rev, line_color=COL["Review"], line_dash="dash", row=1, col=1)
            fig.add_hline(y=t_res, line_color=COL["Restrict"], line_dash="dash", row=1, col=1)
            for st_name in ("Review", "Restrict"):
                sel = g[g["live_state"] == st_name]
                fig.add_trace(go.Scatter(x=sel["timestamp_utc"], y=sel["score"], mode="markers",
                                         name=st_name, marker=dict(color=COL[st_name], size=7)), row=1, col=1)
            fig.add_trace(go.Scatter(x=g["timestamp_utc"], y=g["backlog_truth"], name="backlog",
                                     line=dict(color="#475569", width=1.6)), row=2, col=1)
            fig.add_hline(y=cfg.backlog_threshold, line_color=COL["Restrict"], line_dash="dash", row=2, col=1)
            for h_ in evs["onset_hour"]:
                fig.add_vline(x=ts0_ + pd.Timedelta(hours=int(h_)), line_color="#64748b",
                              line_dash="dot", row="all", col=1)
            fig.update_layout(template="plotly_white", height=460, margin=dict(l=10, r=10, t=40, b=10),
                              legend=dict(orientation="h", y=-0.12))
            st.plotly_chart(fig, width="stretch")

        st.markdown(f'#### {S["log"]}')
        lg = ep_s.sort_values("start_hour")
        log_tbl = pd.DataFrame({
            S["log_t"]: [(ts0_ + pd.Timedelta(hours=int(h))).strftime("%d.%m %H:%M") for h in lg["start_hour"]],
            S["log_c"]: lg["component_id"],
            S["log_l"]: lg["level"].map({1: "Review", 2: "Restrict"}),
            S["log_a"]: lg["level"].map({1: S["act_r"], 2: S["act_s"]}),
            S["log_h"]: lg["n_hours"],
            S["log_r"]: np.where(lg["useful"] == 1, S["useful"], S["false"]),
            S["col_costs"]: lg["cost"].map(fmt_money),
        })
        st.dataframe(log_tbl, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# 6. Данные и дрейф
# ---------------------------------------------------------------------------
with tabs[5]:
    summ = data["summary"]
    d1, d2 = st.columns([1, 2])
    with d1:
        st.markdown(f'#### {TX["dq_defects"]}')
        dq = summ.get("dq_defects", {})
        st.metric(TX["dq_rows"], f'{summ.get("dq_defect_rows", 0):,}'.replace(",", " "))
        if dq:
            st.dataframe(pd.DataFrame({"type": list(dq.keys()), "rows": list(dq.values())}),
                         hide_index=True, width="stretch")
    with d2:
        st.markdown(f'#### {TX["psi_title"]}')
        st.caption(TX["psi_note"])
        psi = data["psi"].sort_values("psi", ascending=True)
        colors = np.where(psi["psi"] < 0.1, "#16a34a", np.where(psi["psi"] < 0.25, "#f59e0b", "#dc2626"))
        fp = go.Figure(go.Bar(
            x=psi["psi"].clip(lower=1e-4), y=psi["feature"], orientation="h", marker_color=colors,
            text=[f"{v:.4f}" for v in psi["psi"]], textposition="outside", cliponaxis=False,
            textfont=dict(size=12, color="#0f172a")))
        fp.add_vline(x=0.1, line_dash="dash", line_color="#f59e0b",
                     annotation_text="0.1", annotation_position="top")
        fp.add_vline(x=0.25, line_dash="dash", line_color="#dc2626",
                     annotation_text="0.25", annotation_position="top")
        fp.update_xaxes(type="log", range=[-4, np.log10(0.6)], title="PSI (log scale)",
                        tickfont=dict(size=13), title_font=dict(size=13))
        fp.update_yaxes(tickfont=dict(size=13))
        fp.update_layout(template="plotly_white", height=max(480, 26 * len(psi) + 80),
                         margin=dict(l=10, r=60, t=30, b=10), font=dict(size=13))
        st.plotly_chart(fp, width="stretch")
        st.caption(f'{TX["psi_range"]}: {data["psi"]["psi"].min():.4f} — {data["psi"]["psi"].max():.4f}')
        with st.expander(TX["feat_title"], expanded=False):
            desc = []
            for f in data["psi"]["feature"]:
                d_, g_ = feat_description(f, L)
                desc.append({TX["feat_name"]: f, TX["feat_group"]: g_, TX["feat_what"]: d_,
                             "PSI": round(float(data["psi"].set_index("feature").loc[f, "psi"]), 4)})
            st.dataframe(pd.DataFrame(desc).sort_values("PSI", ascending=False), hide_index=True, width="stretch")
            st.markdown(TX["feat_note"])

    st.markdown(f'#### {TX["scen_title"]}')
    scen = data["scen"].copy()
    scen_show = pd.DataFrame({
        TX["scen_col"]: scen["scenario"],
        TX["scen_events"]: scen["events"],
        TX["scen_caught"]: scen["caught"],
        TX["scen_recall"]: scen["recall"].map(lambda v: f"{v:.0%}"),
        TX["scen_lead"]: scen["median_lead_h"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—"),
    })
    st.dataframe(scen_show, hide_index=True, width="stretch")

    st.markdown(f'#### {TX["imp_title"]}')
    imp = data["imp"].head(12).sort_values("importance_mean")
    fi = go.Figure(go.Bar(x=imp["importance_mean"], y=imp["feature"], orientation="h",
                          error_x=dict(type="data", array=imp["importance_std"]), marker_color=NAVY))
    fi.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fi, width="stretch")

# ---------------------------------------------------------------------------
# 6. Методология
# ---------------------------------------------------------------------------
with tabs[6]:
    st.markdown(f'#### {TX["method_title"]}')
    st.markdown(TX["method_md"])
    st.caption(f'config {frozen["config"]["config_version"]} · {chosen} · '
               f'{summ["hours"]} h · {summ["components"]} · '
               f'events {summ["events_total"]}')
