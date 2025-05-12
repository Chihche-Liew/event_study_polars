# source code: # https://wrds-www.wharton.upenn.edu/pages/wrds-research/applications/python-replications/programming-python-wrds-event-study/
import os
import json
import wrds
import polars as pl
import numpy as np
from statsmodels.api import OLS, add_constant
import duckdb as db
from polars import col
from tqdm.notebook import tqdm


class EventStudy:
    def __init__(self, output_path='', wrds_username=''):
        self.output_path = output_path or os.path.expanduser('~')
        self.wrds_username = wrds_username

    def connect(self):
        return wrds.Connection(wrds_username=self.wrds_username)

    def eventstudy(self,
                   data=None,
                   model='madj',
                   estwin=250,
                   gap=10,
                   evtwins=-10,
                   evtwine=10,
                   minval=100,
                   output='df'):

        estwins = estwin + gap + abs(evtwins)
        estwine = gap + abs(evtwins) + 1
        evtrang = abs(evtwins) + evtwine + 1
        evtwinx = estwins + 1

        evts = data or [{"edate": "2012-05-29", "cusip": "10002"}]
        params = {'estwins': estwins, 'estwine': estwine, 'evtwins': evtwins, 'evtwine': evtwine, 'evtwinx': evtwinx, 'evtdata': json.dumps(evts)}
        sql = f"""
        SELECT
                a.*,
                x.*,
                c.date as rdate,
                c.ret as ret1,
                (f.mktrf+f.rf) as mkt,
                f.mktrf,
                f.rf,
                f.smb,
                f.hml,
                f.umd,
                (1+c.ret)*(coalesce(d.dlret,0.00)+1)-1-(f.mktrf+f.rf) as exret,
                (1+c.ret)*(coalesce(d.dlret,0.00)+1)-1 as ret,
                case when c.date between a.estwin1 and a.estwin2 then 1 else 0 end as isest,
                case when c.date between a.evtwin1 and a.evtwin2 then 1 else 0 end as isevt,
                case
                  when c.date between a.evtwin1 and a.evtwin2 then (rank() OVER (PARTITION BY x.evtid ORDER BY c.date)-%(evtwinx)s)
                  else (rank() OVER (PARTITION BY x.evtid ORDER BY c.date))
                end as evttime,
                case
                  when c.date = a.date then 1
                  else 0
                end as evtflag
        FROM
          (
            SELECT
              date,
              lag(date, %(estwins)s ) over (order by date) as estwin1,
              lag(date, %(estwine)s )  over (order by date) as estwin2,
              lag(date, %(evtwins)s )  over (order by date) as evtwin1,
              lead(date, %(evtwine)s )  over (order by date) as evtwin2
            FROM crsp_a_stock.dsi
          ) as a
        JOIN
        (select
                to_char(x.edate, 'ddMONYYYY') || x.cusip::text as evtid,
                x.cusip,
                x.edate
        from
        json_to_recordset('%(evtdata)s') as x(edate date, cusip text)
        ) as x
          ON a.date=x.edate
        JOIN crsp_a_stock.dsf c
            ON x.cusip=c.cusip
            AND c.date BETWEEN a.estwin1 and a.evtwin2
        JOIN ff_all.factors_daily f
            ON c.date=f.date
        LEFT JOIN crsp_a_stock.dsedelist d
            ON x.cusip=d.cusip
            AND c.date=d.dlstdt
        WHERE f.mktrf is not null
        AND c.ret is not null
        ORDER BY x.evtid, x.cusip, a.date, c.date
        """ % params
        pdf = self.connect().raw_sql(sql)
        df = pl.from_pandas(pdf)

        df = df.with_columns([
            col('edate').cast(pl.Date),
            col('rdate').cast(pl.Date)
        ])

        def process_grp(grp):
            grp = grp.sort('rdate')
            est = grp.filter(col('isest') == 1)
            evt = grp.filter(col('isevt') == 1)

            if est.height < minval or evt.height != evtrang or evt.filter(col('evtflag') == 1).height < 1:
                return pl.DataFrame()

            alpha = 0.0
            rmse = 1.0

            if model == 'madj':
                alpha = est.select(col('exret')).to_numpy().mean()
                rmse = est.select(col('exret')).to_numpy().std(ddof=1)
                evt = evt.with_columns([
                    pl.lit(alpha).alias('INTERCEPT'),
                    pl.lit(rmse).alias('RMSE'),
                    pl.lit(alpha).alias('alpha'),
                    col('exret').alias('abret'),
                    col('mkt').alias('expret')
                ])
            else:
                est_pd = est.to_pandas()
                y = est_pd['ret']
                if model == 'm':
                    X = add_constant(est_pd[['mktrf']])
                elif model == 'ff':
                    X = add_constant(est_pd[['mktrf','smb','hml']])
                elif model == 'ffm':
                    X = add_constant(est_pd[['mktrf','smb','hml','umd']])
                res = OLS(y, X).fit()
                params = res.params.to_dict()
                alpha = params['const']
                betas = {k: v for k, v in params.items() if k != 'const'}
                rmse = np.sqrt(res.mse_resid)

                expr = pl.lit(alpha)
                for f_name, b_val in betas.items():
                    expr = expr + pl.lit(b_val) * col(f_name)
                evt = evt.with_columns([
                    pl.lit(alpha).alias('INTERCEPT'),
                    pl.lit(rmse).alias('RMSE'),
                    pl.lit(alpha).alias('alpha'),
                    expr.alias('expret'),
                    (col('ret') - expr).alias('abret')
                ])

            def compute_cret(ret_series):
                result = []
                acc = 0.0
                for r in ret_series:
                    tmp = (r * acc) + (r + acc)
                    acc = tmp
                    result.append(tmp)
                return pl.Series(result)

            def compute_cexpret(expret_series):
                result = []
                acc = 0.0
                for r in expret_series:
                    tmp = (r * acc) + (r + acc)
                    acc = tmp
                    result.append(tmp)
                return pl.Series(result)

            evt = evt.sort('date')
            evt = evt.with_columns([
                compute_cret(evt['ret']).alias('cret'),
                compute_cexpret(evt['expret']).alias('cexpret'),
                col('abret').cum_sum().alias('car'),
                pl.lit((est.height - 2) / (est.height - 4)).alias('pat_scale')
            ])

            evt = evt.with_columns([
                (col('abret') / rmse).cum_sum().alias('sar'),
                (col('cret') - col('cexpret')).alias('bhar'),
                (col('car') / np.sqrt(evtrang * rmse**2)).cum_sum().alias('scar')
            ])
            return evt

        keys = df.select(['cusip','edate']).unique().to_pandas().to_dict(orient='records')
        result_list = []
        for evt in tqdm(keys):
            grp = df.filter((col('cusip')==evt['cusip']) & (col('edate')==evt['edate']))
            out = process_grp(grp)
            if out.height > 0:
                result_list.append(out)
        processed = pl.concat(result_list) if result_list else pl.DataFrame()
        df_evt = processed.drop_nulls().to_pandas()

        df_stats = db.query(f"""
            SELECT evttime,
                   AVG(car) AS car_m,
                   AVG(ret) AS ret_m,
                   AVG(abret) AS abret_m,
                   STDDEV_SAMP(abret) AS abret_v,
                   AVG(sar) AS sar_m,
                   STDDEV_SAMP(sar) AS sar_v,
                   AVG(scar) AS scar_m,
                   STDDEV_SAMP(scar) AS scar_v,
                   AVG(bhar) AS bhar_m,
                   COUNT(*) AS n,
                   AVG(cret) AS cret_edate_m,
                   AVG(car) AS car_edate_m,
                   AVG(bhar) AS bhar_edate_m
            FROM df_evt
            GROUP BY evttime
            ORDER BY evttime
        """
        ).df()

        df_window = df_evt[['cusip','edate','rdate','evttime','ret','abret']].sort_values(['cusip','evttime'])
        max_t = df_evt['evttime'].max()
        df_date = df_evt[df_evt['evttime'] == max_t][['cusip','edate','cret','car','bhar']]
        df_date = df_date.sort_values(['cusip','edate'])

        if output == 'df':
            return {'event_stats': df_stats,
                    'event_window': df_window,
                    'event_date': df_date}
