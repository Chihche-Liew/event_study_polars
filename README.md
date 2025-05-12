# event_study_polars
Replication of https://wrds-www.wharton.upenn.edu/pages/wrds-research/applications/python-replications/programming-python-wrds-event-study/?no_login_redirect=True
Using polars and duckdb to accelerate the original version on WRDS.

requirements:
- python==3.12.7
- wrds==3.2.0
- polars==1.27.1
- numpy==1.26.4
- statsmodels==0.14.4
- duckdb==1.1.3
- tqdm==4.66.5

test environment:
AMD Ryzen 7 9700X; 64GB RAM
With ~12000 events, takes 1 min on average rather than > 4 hrs for original version.
