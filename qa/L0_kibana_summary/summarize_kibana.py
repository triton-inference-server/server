#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import query_kibana
import pandas
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt


# convert string timestamp to datatime object
def StringToDateTime(timestamp_str):
    return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")


# convert datatime object to string timestamp
def DateTimeToString(dtime):
    return dtime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


# convert string timestamp to Y-M-D string
def TimestampToYDMString(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")


# timestamp,onnx,libtorch,netdef,custom,savedmodel,graphdef
def create_timestamp_dataframe(df, metric, unique_backends):
    x = pandas.DataFrame(columns=['timestamp'] + unique_backends)
    d = dict()
    time_list = []
    for i, row in df.iterrows():
        if row['backend'] in d:
            # average out timestamps
            d['timestamp'] = sum(time_list) / len(time_list)
            x = x.append(d, ignore_index=True)
            d = dict()
            time_list = []
        d[row['backend']] = row[metric]
        time_list.append(StringToDateTime(row['timestamp']).timestamp())

    return x


# Smoothen to YYYY-MM-DD
def create_date_dataframe(x, unique_backends):
    df_days = pandas.DataFrame(columns=['date'] + unique_backends)
    old_ymd = None
    curr_i = 0
    same_day = 1
    for i, row in x.iterrows():
        current_ymd = TimestampToYDMString(row['timestamp'])
        if old_ymd is None:
            old_ymd = current_ymd
        elif old_ymd != current_ymd:
            val_list = dict()
            for col_i in range(1, len(unique_backends) + 1):
                val_list[unique_backends[col_i - 1]] = round(
                    x.iloc[i - same_day:i, col_i].mean(), 2)
            val_list['date'] = old_ymd
            df_days = df_days.append(val_list, ignore_index=True)
            curr_i += 1
            old_ymd = current_ymd
            same_day = 1
        else:
            same_day += 1
    return df_days


# Create moving average
def create_moving_average_dataframe(df_days, unique_backends, ma_days=7):
    ma_df = pandas.DataFrame(columns=['date'] + unique_backends)
    for i in range(df_days.shape[0] - ma_days):
        val_list = dict()
        for col in range(1, len(unique_backends) + 1):
            val_list[unique_backends[col - 1]] = round(
                df_days.iloc[i:i + ma_days, col].mean(), 2)
        val_list['date'] = df_days.iloc[i + ma_days]['date']
        ma_df = ma_df.append(val_list, ignore_index=True)
    return ma_df


# Create current moving average
def current_moving_average_dataframe(metric,
                                     value_list,
                                     where_dict,
                                     last_date=None,
                                     plot=False,
                                     plot_file="plot.png"):
    if last_date is None:
        today = date.today()
    else:
        today = datetime.strptime(last_date, "%Y-%m-%d").date()

    end_date = today.strftime("%Y-%m-%d")
    start_date = (today - timedelta(days=120)).strftime("%Y-%m-%d")
    qk = query_kibana.QueryKibana()
    rows = qk.fetch_results(value_list,
                            where_dict=where_dict,
                            limit=0,
                            start_date=start_date,
                            end_date=end_date)
    qk.close()
    df = pandas.DataFrame(rows, columns=[metric, 'backend', 'timestamp'])
    df = df.sort_values('timestamp')
    unique_backends = list(set(df['backend']))
    timestamp_df = create_timestamp_dataframe(df, metric, unique_backends)
    date_df = create_date_dataframe(timestamp_df, unique_backends)

    ma_df = pandas.DataFrame(columns=['days'] + unique_backends)
    for d in (120, 60, 30, 7):
        start_date = (today - timedelta(days=d)).strftime("%Y-%m-%d")
        date_subset = date_df[date_df['date'] > start_date]
        val_list = dict()
        for col in range(1, len(unique_backends) + 1):
            val_list[unique_backends[col - 1]] = round(
                date_subset.iloc[:, col].mean(), 2)
        val_list['days'] = d
        ma_df = ma_df.append(val_list, ignore_index=True)

    if plot:
        plt.switch_backend('agg')
        ma_df.drop(columns=['days']).plot()
        plt.xticks([0, 1, 2, 3], ma_df.days)
        lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)
        plt.xlabel('Moving average days')
        plt.ylabel(metric.capitalize())
        plt.savefig(plot_file, bbox_extra_artists=(lgd,), bbox_inches='tight')

    return ma_df
