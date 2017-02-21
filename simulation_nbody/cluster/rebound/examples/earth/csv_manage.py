import numpy as np
import pandas as pd
import rebound
from os import listdir as ls


def read_hash(hash_file='hash.csv'):
    hash_df = pd.read_csv(hash_file, index_col=0,
                          dtype={'Mass': float,
                                 'Radius': float})
    return hash_df


def get_wd_hash():
    wd_hash = rebound.hash('wd').value
    return wd_hash


def get_N_particles(hash_df):
    return hash_df.shape[0]


def read_time_step(month_num, N, pos_count, month_suffix='_months.csv'):
    skip_count = (pos_count)*(N+1)
    use_count = N
    month_csv = str(month_num) + month_suffix
    step_df = pd.read_csv(month_csv, index_col=0, skiprows=skip_count,
                          nrows=use_count, dtype=float)
    hash_df = read_hash()
    step_df['Radius'] = pd.Series([hash_df[' Radius'][hash_temp]
                                   for hash_temp in step_df.index],
                                  index=step_df.index)
    step_df['Mass'] = pd.Series([hash_df[' Mass'][hash_temp]
                                 for hash_temp in step_df.index],
                                index=step_df.index)
    return step_df


def num_time_steps(month_num, N, month_suffix='_months.csv'):
    month_csv = str(month_num) + month_suffix
    month_csv = pd.read_csv(month_csv, usecols=[0])
    return (len(month_csv.index))/(N+1)


def get_month_num(month_suffix='_months.csv'):
    file_list = ls('.')
    month_csvs = [f_name for f_name in file_list
                  if f_name.endswith(month_suffix)]
    month_nums = [int(str(month_file[:-len(month_suffix)]))
                  for month_file in month_csvs]

    return np.max(month_nums), month_nums, month_csvs


if __name__ == '__main__':
    hash_df = read_hash()
    N = get_N_particles(hash_df)
    print get_month_num()
