# WRDS Event Study (Accelerated with Polars + DuckDB)

Replication of https://wrds-www.wharton.upenn.edu/pages/wrds-research/applications/python-replications/programming-python-wrds-event-study/?no_login_redirect=True using **Polars** and **DuckDB** to accelerate the original version on WRDS.

## Features
- WRDS-based SQL query for CRSP + Fama-French data

- Supports multiple models: market-adjusted, CAPM, Fama-French 3-factor, and Carhart 4-factor

- Computes abnormal returns (`abret`), cumulative abnormal returns (`car`), and `bhar`

- Efficient in-memory processing using Polars and DuckDB

- ~240x speedup on large-scale event samples (~12,000 events)

## Tested Environment
- CPU: AMD Ryzen 7 9700X
- RAM: 128 GB
- Python: 3.12
- WRDS Access: Required

## Dependencies
Recommended packages for Python ≥ 3.10:
```
pip install wrds polars numpy statsmodels duckdb tqdm
```
Minimal `conda` environment:
```
conda create -n eventstudy python=3.10
conda activate eventstudy
pip install wrds polars numpy statsmodels duckdb tqdm
```

## Usage
```{python}
from event_study_polars import EventStudy

es = EventStudy(wrds_username='your_wrds_id')
results = es.eventstudy(data=[{"edate": "2012-05-29", "cusip": "10002"}],
                        model='ff',  # options: 'madj', 'm', 'ff', 'ffm'
                        estwin=250,
                        evtwins=-10,
                        evtwine=10,
                        gap=10)
```
Returns a dictionary of:
- event_stats: Time-series stats across events

- event_window: Raw abnormal returns by event time

- event_date: CAR/BHAR on event date

## Citation
Adapted from: https://wrds-www.wharton.upenn.edu/pages/wrds-research/applications/python-replications/programming-python-wrds-event-study/
