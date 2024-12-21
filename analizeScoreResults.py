#%% データの読み込み
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

dataset = pd.read_csv("result/scored_dataset.csv")

# %% スコアのトップ10を表示
# グラフの値はスコア、バー上に表示される文字は実際の値です。

import matplotlib.cm as cm
tag_cols = [
    "val_monthly_fee",
    "val_area",
    "val_walk_time",
    "val_build_age",
    "commute_time",
    "diff",
]
tag_cols_jap = [
    "家賃",
    "面積",
    "徒歩距離",
    "築年数",
    "通勤時間",
    "割安感",
]
tag_cols_score = [tag_col + "_score" for tag_col in tag_cols]

color_list = [cm.winter_r(i / len(tag_cols)) for i in range(len(tag_cols))]
dataset_top_score = dataset.sort_values("Total_score", ascending=False).iloc[:10]
for i in range(len(dataset_top_score)):
    plt.figure(figsize=(12, 5))
    plt.bar(tag_cols_jap, dataset_top_score.iloc[i][tag_cols_score], color=color_list)
    for x, y, y_text in zip(
        tag_cols_jap, dataset_top_score.iloc[i][tag_cols_score], dataset_top_score.iloc[i][tag_cols]
    ):
        plt.text(x, y, round(y_text, 2), ha="center", va="bottom", fontsize=16)
        plt.hlines(0, -0.5, 5.5, color="gray")
        plt.ylim([-0.5, 1.0])
        plt.title(
            dataset_top_score.iloc[i]["名称"] + "({})".format(dataset_top_score.iloc[i]["区"]),
            fontsize=16,
        )
        plt.xticks(fontsize=12)
    plt.savefig('result/top_score.png')
    plt.close()


#%% 区ごとの傾向分析~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from tools import show_bar
val_columns = [col for col in dataset.columns if 'val' in col]
val_columns = [col for col in val_columns if 'score' not in col]
val_columns.append('区')
df_group_mean = dataset[val_columns].groupby('区').mean()
df_group_std = dataset[val_columns].groupby('区').std()
for tag_col in df_group_mean.columns:
    x = df_group_mean.index
    y = df_group_mean[tag_col]
    yerr = df_group_std[tag_col]
    show_bar(x, y, yerr, tag_col)

#%% 路線ごとの傾向分析~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print(dataset["nearest_line"].unique())
dataset_ext = dataset[
    (dataset["nearest_line"] == "西武池袋線")
    | (dataset["nearest_line"] == "西武三田線")
]
# 分析スクリプト未実装

#%% 築年数による家賃の変化を確認~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df_result_new = dataset[
    (dataset["val_build_age"] == 0)
    ]
df_result_234 = dataset[
    ((dataset["val_build_age"] >= 1) & (dataset["val_build_age"] <= 4))
    ]
df_result_u10 = dataset[
    ((dataset["val_build_age"] >= 5) & (dataset["val_build_age"] <= 10))
    ]
df_result_u20 = dataset[
    ((dataset["val_build_age"] >= 11) & (dataset["val_build_age"] <= 20))
    ]

print(df_result_new["val_monthly_fee"].mean())
print(df_result_234["val_monthly_fee"].mean())
print(df_result_u10["val_monthly_fee"].mean())
print(df_result_u20["val_monthly_fee"].mean())
df_combined = pd.DataFrame({'新築': df_result_new["val_monthly_fee"], 
                            '築2~4年': df_result_234["val_monthly_fee"],
                            '築5~10年': df_result_u10["val_monthly_fee"],
                            # '築11～20年': df_result_u20["val_monthly_fee"]
                            })
plt.hist(df_combined, label=df_combined.columns ,bins=20)
plt.legend()
plt.savefig('result/fee_variation.png')
plt.close()
