# Payment Network Capacity Risk & Early Warning

A portfolio prototype of a decision-support system for early detection of channel overload in a P2P payment network, and for evaluating the economics of responding to warnings.

**Live demo:** https://payment-network-capacity-risk-emrarknxd6tnubgccdjpwd.streamlit.app
The dashboard switches between English, Українська and Русский (see the selector in the sidebar).
**Русская версия README:** [README_ru.md](README_ru.md)

> **Important.** All data is synthetic. The project demonstrates a methodology and an analytical approach. The results are not evidence of effectiveness on real data and do not reproduce the internal algorithms of banks or payment providers.

---

## Idea

A payment network can be viewed as a system of channels: when one channel gets clogged, the capacity of the whole network drops. The goal of the project is to notice an overload of a channel early, from observed load dynamics, and to decide whether a response is worth it.

A response costs money (checking a channel, moving part of the flow), and a missed overload costs money too. Therefore the alert threshold is chosen by economics, not by a classification metric.

The project is considered at two levels:

1. **Account Availability Risk Score**: a risk score for restriction of an individual account's availability, based on transaction dynamics. *Concept only: not implemented in the current version.*
2. **Payment Network Capacity Risk & Early Warning**: assessment of network operational capacity and early warning of channel overloads. *Implemented.*

The link between the levels (account restriction → network overload) is a hypothesis that the project does not yet test.

---

## What is implemented

- **Synthetic data.** Hourly simulation of six channels over three years: demand, capacity, queue. Overload emerges from queue dynamics rather than being set directly. Four stress types (gradual build-up, cascade, flash surge, short surge without overload), plus a precursor without overload used in the scenario. Telemetry includes defects: missing values, delays and distortions in 0.25% of rows.
- **Features.** Lags, rolling windows (6, 12, 24 hours), trends of demand, capacity and queue, channel utilisation, time to clear the queue, a data-quality flag. Features use only past and current observations.
- **Models.** Four approaches: a no-skill constant, a queue-based heuristic, logistic regression, and gradient boosting (HistGradientBoosting). The target is whether an overload starts within the next 6 hours. Probabilities are calibrated with Platt scaling on a separate block.
- **Temporal validation.** Training, model validation, calibration, threshold selection and testing follow time order, with no random shuffling. Blocks are separated by an embargo of 12 hours, and labels are purged at block boundaries (6 hours). Hours already in overload are excluded from evaluation.
- **Thresholds and response levels.** Two levels:
  - **Review**: check the backup channel and prepare to move part of the flow;
  - **Restrict**: switch 30% of the flow to a backup processor.

  Both are recommendations to an operator; there is no automatic rerouting. Thresholds are chosen on a separate block by net benefit under two limits: the number of alert episodes per day and the share of hours in alert mode. The test block is not used for selection.
- **Decision economics.** The model accounts for the loss from an event without warning, the cost of a check, the cost of a restriction, the hourly monitoring cost, and the share of loss avoided by each level. For each level, the break-even episode precision is computed.
- **Scenario analysis.** A designed stress in two channels. Three policies on the same scenario: no system, Review only, Review + Restrict. Channel states come from the trained model and thresholds, not from a hand-written table.
- **Interactive dashboard.** Seven tabs: Overview, Channel, Models, Economics, Scenario, Data & drift, Methodology. The interface is available in English, Ukrainian and Russian. All figures are read from the pipeline output or recomputed by the same core as the pipeline. The model to view is selected in the sidebar. Switching changes the thresholds, overview, channel, economics and scenario, but not the model chosen for decisions. A model for which no admissible thresholds exist is marked in the table, and its utility and economic metrics are not reported.

---

## Results

Test period: 162 days, 26 events. Values in brackets are 95% intervals (bootstrap over events and episodes). The intervals are wide because there are few events.

| Metric | Value |
|---|---|
| Recall (events with a warning) | 92% (81–100%) |
| Median lead time before an event | 4 h (3–5) |
| Alert episode precision | 22% (14–29%) |
| Net benefit, conventional units | 1.22 M (1.01–1.41 M) |
| System ROI | 3.2× (2.6–4.0) |
| ROC-AUC: logistic regression / gradient boosting | 0.86 / 0.81 |
| Brier skill vs. constant: logistic regression / gradient boosting | 0.31 / 0.26 |

**Model selection.** The decision model is gradient boosting. It was chosen by net benefit on the threshold block, under a rule fixed before the test set was examined. On the test set, logistic regression scored higher on ROC-AUC and on economics. The dashboard reports this openly. The conclusion: the more complex model did not demonstrate an advantage here. The Models tab includes an 'Alert hours' column: the share of hours in which a channel is in alert mode. For the queue rule it is 100%: the alert never switches off, so its 100% utility is formal and does not indicate warning quality.

**Calibration.** Weak. At low predicted probabilities, the predicted risk does not match the observed frequency, and in the top decile the model underestimates risk (3.5% predicted vs. 4.7% observed). The probabilities should be read as a relative risk ranking, not as exact probabilities.

**How to read these numbers.**
- *Recall*: the share of events preceded by at least one signal within 6 hours before they started.
- *Episode precision*: the share of alert episodes followed by an actual event within their time window or the following hours. An episode merges signals separated by pauses of up to 3 hours.
- *Break-even precision*: the minimum precision at which a response level pays for itself. For Restrict it is about 11%, the observed value is 81%. For Review it is about 4.5%, the observed value is 12%. Review pays off, but only just, and it produces many false alarms.
- *Intervals* are wide because the test set has 26 events. Any conclusion should be read with that caveat.

---

## Decision scenario

The pipeline applies a designed stress to two channels:

- **COMP-06**: a load build-up without overload. The system issues a Review, which is a false alarm. The cost of the check is included in the economics.
- **COMP-04**: a build-up that turns into an overload. The Restrict level fires 4 hours before the overload and protects 75 k of the 100 k conventional units of loss.

| Policy | Loss avoided | Costs | Net benefit |
|---|---|---|---|
| No system | 0 | 0 | 0 |
| Review only | 55,000 | 24,200 | 30,800 |
| Review + Restrict | 75,000 | 29,700 | 45,300 |

The Restrict level adds 14.5 k over the Review-only policy. The channel states in the scenario come from the model, not from a hand-written table. An episode that has not finished by the end of the window is not included in costs, because its outcome is unknown.

**Caveat.** Restrict fires near the upper end of the calibrated risk range (threshold 0.44, maximum about 0.45), so in the scenario it lasts only a few hours before the overload.

---

## Limitations

- The data is synthetic. Conclusions about industrial effectiveness should not be drawn.
- The account-level component (Account Availability Risk Score) is not implemented in this version.
- The link "account restriction → network overload" is neither modelled nor tested.
- The test set has 26 events: the intervals are wide and the conclusions are preliminary.
- Calibration is imprecise at low probabilities.
- Costs, losses and loss-avoidance shares are assumptions, not measurements.
- The synthetic world is stationary, so feature drift (PSI) is stable by construction. Queue features are almost always zero, so their PSI is formally zero.

---

## Repository structure

```
├── capacity_core.py           shared core: configuration, episodes, thresholds, economics, bootstrap
├── run_capacity_pipeline.py   simulation, features, models, calibration, thresholds, scenario, export
├── dashboard_capacity.py      interactive dashboard (Streamlit)
├── capacity_output/           output of the latest run, read by the dashboard
├── requirements.txt           dependencies
├── .gitignore
├── README.md                  this file (English)
└── README_ru.md               Russian version
```

The pipeline and the dashboard share one core, `capacity_core.py`, so the numbers in reports and on screen are identical.

---

## Methodology

1. **Event**: an interval during which a channel's queue is above the threshold of 220 conventional units. Its start is the moment of the event.
2. **Label**: whether an event starts within the next 6 hours.
3. **Time split**: training 55%, model validation 10%, calibration 10%, threshold selection 10%, test the remainder (about 15%). A 12-hour gap separates the blocks.
4. **Thresholds.** Review and Restrict are chosen on the threshold block by net benefit. Constraints: at most one alert episode per day, and at most 3% of hours in alert mode. If no admissible thresholds exist under these limits, the model gets no utility or economic metrics.
5. **Alert episode**: a run of hours with risk above the Review threshold, merged across pauses of up to 3 hours. An episode is useful if an event starts within its time span or in the 6 hours after it.
6. **Event avoided**: a signal was present in the 6 hours before the event started. Share of loss avoided: 55% for Review, 75% for Restrict.
7. **Economic assumptions** (set in `Config`): loss from an event without warning 100,000; check 2,500 per episode; restriction 8,000 per episode; monitoring 150 per alert hour.
8. **Uncertainty** is estimated by bootstrap (300 iterations), resampling events and episodes separately.

The "Methodology" tab of the dashboard describes the same points.

---

## Running

Python 3.11 or newer is required.

```bash
pip install -r requirements.txt
python run_capacity_pipeline.py      # about 30 seconds, writes capacity_output/
streamlit run dashboard_capacity.py
```

The `capacity_output/` folder is already in the repository, so the dashboard runs without the pipeline. Run the pipeline if you want to recompute the results or change the `Config` parameters (`capacity_core.py`) or the stress schedule `SCENARIO_SCHEDULE` (`run_capacity_pipeline.py`).

Economic parameters (loss, costs, response effectiveness) can be changed in the dashboard sidebar without rerunning the pipeline.

**Reproducibility.** Generator seed 20251029. Results are deterministic for the same library versions. The run configuration is saved in `capacity_output/config.json`.

---

## What's next

- Implementation of the account-level component (Account Availability Risk Score) and a test of its link to network overload.
- Validation on real or semi-real data, if access becomes possible.
- Better calibration at low probabilities.
- A longer test and an assessment of threshold stability.

---

## License

MIT. See the `LICENSE` file.

---

## Author

**Vasyl Yanov (Василий Янов)**: Lead Data Scientist / Risk Analytics Architect.

Financial and data analyst, PhD in systems analysis and cybernetics. Specialisation: risk modelling, decision economics and early-warning systems for fintech and payment platforms.

LinkedIn: https://www.linkedin.com/in/vasyl-yanov-10981a31b/

I welcome feedback on the architecture, the methodology and the approach to decision economics. Please open an issue in the repository for comments.
