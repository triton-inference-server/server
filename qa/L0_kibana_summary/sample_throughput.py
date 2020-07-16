import query_kibana
import pandas
from datetime import datetime

qk = query_kibana.QueryKibana()
value_list = ["d_infer_per_sec", "s_framework", "\'@timestamp\'"]
where_dict = {"s_shared_memory": "none","s_benchmark_name": "nomodel", "l_size": "4194304", "s_protocol": "http"}
rows = qk.fetch_results(value_list, where_dict=where_dict, limit=0)
qk.close()

df = pandas.DataFrame(rows, columns=['throughput', 'backend', 'timestamp'])
print(df.head())
unique_backends = list(set(df['backend']))

# convert string timestamp to datatime object
def StringToDateTime(timestamp_str):
    return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")

def FromDateTime(dtime):
    return dtime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

def TimestampToYDMString(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

# timestamp,onnx,libtorch,netdef,custom,savedmodel,graphdef
def create_timestamp_dataframe(df):
    x = pandas.DataFrame(columns=['timestamp'] + unique_backends)
    d = dict()
    time_list = []
    for i, row in df.iterrows():
        if row['backend'] in d:
            # average out timestamps
            d['timestamp'] = sum(time_list)/len(time_list)
            x = x.append(d, ignore_index=True)
            d = dict()
            time_list = []
        d[row['backend']] = row['throughput']
        time_list.append(StringToDateTime(row['timestamp']).timestamp())

    return x

x = create_timestamp_dataframe(df)
# Save to csv
x.to_csv("throughput_p4194304_http_nomodel.csv", index=False)

# Smoothen to YYYY-MM-DD
def create_date_dataframe(x):
    df_days = pandas.DataFrame(columns=['date'] + list(set(df['backend'])))
    old_ymd = None
    curr_i = 0
    same_day = 1
    for i, row in x.iterrows():
        current_ymd = TimestampToYDMString(row['timestamp'])
        if old_ymd is None:
            old_ymd = current_ymd
        elif old_ymd != current_ymd:
            val_list = dict()
            for col_i in range(1, len(unique_backends)+1):
                val_list[unique_backends[col_i-1]] = round(x.iloc[i-same_day:i, col_i].mean(), 2)
            val_list['date'] = old_ymd
            df_days = df_days.append(val_list, ignore_index = True)
            curr_i += 1
            old_ymd = current_ymd
            same_day = 1
        else:
            same_day += 1
    return df_days

df_days = create_date_dataframe(x)
# Save to csv
df_days.to_csv("throughput_p4194304_http_nomodel_d1.csv", index=False)

# def calc_moving_avg(df, i, n_days, col):
#     return (df.iloc[i:i+n_days, col]).sum(axis=0)

# # Create Moving average
# moving_average_days = 7
# for i in range(0, x.shape[0]-moving_average_days):
#     x.loc[df.index[i+2], 'MA_3'] = np.round(
#         ((df.iloc[i, 1] + df.iloc[i+1, 1] + df.iloc[i+2, 1])/3), 1)
