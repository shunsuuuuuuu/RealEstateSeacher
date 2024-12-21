import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

def show_bar(x, y, yerr, tag_col):
    y_min = y.min() - yerr.max()*1.2
    y_max = y.max() + yerr.max()*1.2

    plt.figure(figsize=(12, 8))
    plt.bar(x, y, yerr=yerr, capsize=10, color="lightblue")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(tag_col, fontsize=20)
    plt.ylim([y_min, y_max])
    if y_min == 0:
        plt.ylim([0, y_max])
    plt.savefig("result/" + tag_col + ".png")
    plt.close()

# def get_trans_detail(trans_detail):
#     station_list = []
#     time_list = []
#     for trans_sec in trans_detail:
#         trans_station = re.findall("\d*:\d*", trans_sec)[0]
#         pattern = re.compile(r"\d")
#         trans_time = pattern.split((trans_sec))[-1]
#         station_list.append(trans_station)
#         time_list.append(trans_time)
#     df = pd.DataFrame(station_list, time_list).reset_index()
#     data = np.array([station_list, time_list])
#     df.columns = ["Station", "Time"]
#     return data

def calc_group_ave(data, tag_col, exp_col):
    print("tag_col", tag_col)
    print("exp_col", exp_col)
    ave_list = []
    data[exp_col + "_to_mean"] = [0] * len(data)
    for grp in data[exp_col].unique():
        ave = data[data[exp_col] == grp][tag_col].mean()
        print(grp, "Average:", ave)
        data.loc[data[exp_col] == grp, exp_col + "_to_mean"] = ave
        ave_list.append(ave)
    data_order = pd.DataFrame([data[exp_col].unique(), ave_list]).transpose()
    data_order.columns = [exp_col, "ave"]
    data_order = data_order.sort_values("ave", ascending=False).reset_index(drop=True)
    data_order[exp_col + "_rank"] = data_order.index
    data_rtn = pd.merge(data, data_order[[exp_col, exp_col + "_rank"]], on=exp_col)
    return data_rtn


def col_to_end(df, col_name):
    l_url = df[col_name]
    df = df.drop(col_name, axis=1)
    df[col_name] = l_url
    return df
