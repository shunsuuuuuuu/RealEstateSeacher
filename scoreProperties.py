# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 01:56:59 2022

@author: chibi
"""
# %% ライブラリ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from retry import retry
import requests
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description="Command line argument parser")
parser.add_argument("--dest_station", "-ds", help="Destination station used when commuting")
parser.add_argument("--rent_fee", "-fee", help="Acceptable rent fee")
parser.add_argument("--build_age", "-age", help="Acceptable building age")
parser.add_argument("--floor_area", "-area", help="Minimum floor area")
parser.add_argument("--walk_time", "-t", help="Acceptable walk time from nearest station")
args = parser.parse_args()

if args.dest_station is not None:
    dest_station = args.dest_station
    include_ride_time = True
else:
    dest_station = "東京"
    include_ride_time = False

# %% スクレイピングしたデータの読み込み ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
datasets = pd.read_csv("result/datasets.csv")
print(datasets['間取り'].unique())

# 前処理：　学習モデル用に外れ値のデータ除去
datasets = datasets[datasets["val_monthly_fee"] < 30]
print(datasets.shape)

datasets = datasets[datasets["val_walk_time"] < 20]
print(datasets.shape)

datasets = datasets[datasets["val_area"] < 80]
print(datasets.shape)

datasets = datasets[datasets["val_build_age"] < 30]
print(datasets.shape)

# ダブり物件削除
datasets = datasets.drop_duplicates(subset="URL")
print(datasets.shape)
datasets = datasets.drop_duplicates(subset="名称")
print(datasets.shape)

# 区を説明変数に追加
datasets["区"] = [(i.split("区")[0]).replace("東京都", "") for i in datasets["アドレス"]]

# %% 機械学習のデータ準備 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import lightgbm as lgb  # LightGBM
from sklearn.svm import SVR
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

# 説明変数,目的変数
tag_col = ["val_monthly_fee"]
exp_cols = ["val_walk_time", "val_area", "val_build_age", "val_floor"]

# トレーニングデータ,テストデータの分割
test_size = 0.7
test_index = list(datasets.sample(frac=test_size).index)
train_index = list(set(datasets.index) ^ set(test_index))
X_test = datasets.loc[test_index, :][exp_cols].values
X_train = datasets.loc[train_index, :][exp_cols].values
y_test = np.array((datasets.loc[test_index, :][tag_col].values)).reshape(
    len(test_index),
)
y_train = np.array((datasets.loc[train_index, :][tag_col].values)).reshape(
    len(train_index),
)

# 区の情報をOne-Hotエンコーディングを実装
X_ku_test = pd.get_dummies(datasets.loc[test_index, "区"]).values
X_test = np.hstack([X_test, X_ku_test])
X_ku_train = pd.get_dummies(datasets.loc[train_index, "区"]).values
X_train = np.hstack([X_train, X_ku_train])

# ラベルエンコーディング（OrdinalEncoder）
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# X_ku = le.fit_transform(datasets['区'].values)
# X_ku = X_ku.reshape(len(X_ku), 1)
# X = np.hstack([X, X_ku])

# 自作のエンコーディング (Target ranking encord)：　区ごとに家賃の高い順にランク付けする
# from tools import calc_group_ave
# datasets = calc_group_ave(datasets, tag_col, '区')
# X_ku = datasets["区_rank"].values.reshape(len(datasets), 1)
# X_ku = datasets["区_to_mean"].values.reshape(len(datasets), 1)
# X = np.hstack([X, X_ku])

# %% 学習モデルの作成 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# データを標準化
# sc = StandardScaler()
# sc.fit(X_train) #学習用データで標準化
# X_train = sc.transform(X_train)
# X_test = sc.transform(X_test)

# モデルの学習
# model = LinearRegression() #Linear
# model = SVR(kernel='linear', C=1, epsilon=0.1, gamma='auto') #SVR
model = RandomForestRegressor()
# model = lgb.LGBMRegressor() # LightGBM

if os.path.exists("model.pickle"):
    print("model.pickle exists")
    with open("model.pickle", mode="rb") as f:
        model = pickle.load(f)
else:
    print("Not exists")
    print("Creating model...")
    model.fit(X_train, y_train)
    with open("model.pickle", mode="wb") as f:
        pickle.dump(model, f)

# テストデータの予測
y_pred = model.predict(X_test)

# 真値と予測値の表示
df_pred = pd.DataFrame({"rental_fee": y_test, "rental_fee_pred": y_pred})

# 散布図を描画(真値 vs 予測値)
plt.plot(
    y_test, y_test, color="red", label="x=y"
)  # 直線y = x (真値と予測値が同じ場合は直線状に点がプロットされる)
plt.scatter(y_pred, y_test)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.title("y vs y_pred")
plt.savefig("result/figure/predict/y_vs_y_pred.png")

rmse = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE", rmse)
result = model.score(X_test, y_test)
print("R2", result)


# %% 予測値との差分を評価 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df_exp = datasets.loc[test_index, :].reset_index()
df_pred["diff"] = (
    df_pred["rental_fee_pred"] - df_pred["rental_fee"]
)
datasets_extended = pd.concat([df_exp, df_pred], axis=1)


# %% 条件を好みに絞る ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 次に実行する勤務地までの乗車時間の算出に時間がかかるため、件数を絞ります。

## 家賃20万以下
if args.rent_fee is not None:
    datasets_extended = datasets_extended[datasets_extended["val_monthly_fee"] < float(args.rent_fee)]
## 築年数20年以下
if args.build_age is not None:
    datasets_extended = datasets_extended[datasets_extended["val_build_age"] < float(args.build_age)]
## 間取りは2LDK, 1LDK, 2DK ('3LDK' '2LDK' '1K' '1LDK' 'ワンルーム' '1DK' '2DK' '2K' '3DK' '3SLDK' '1SLDK' '2SLDK' '4DK' '1SK' '4SLDK' より選択)
# datasets_extended = datasets_extended[
#     (datasets_extended["間取り"] == "2LDK")
#     | (datasets_extended["間取り"] == "1LDK")
#     | (datasets_extended["間取り"] == "2DK")
# ]
## 床面積40m2以上
if args.floor_area is not None:
    datasets_extended = datasets_extended[datasets_extended["val_area"] > float(args.floor_area)]
## 徒歩15分以内
if args.walk_time is not None:
    datasets_extended = datasets_extended[datasets_extended["val_walk_time"] < float(args.walk_time)]
## 足立区なんかには住まないよね？
datasets_extended = datasets_extended[datasets_extended["区"] != '足立']

# %% 物件からの通勤時間を導出 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# "get_transfer_info" is a private function wrritten in another file "yahoo_transfer.py"
from yahoo_transfer import get_transfer_info

print('Calculating ride time to the dest_station using Yahoo Transfers API.')
print('The number of propaties : {}'.format(len(datasets_extended)))
datasets_extended["nearest_st"] = [
    (i.split("/")[1].split("駅")[0]) for i in datasets_extended["アクセス"]
]
datasets_extended["nearest_line"] = [(i.split("/")[0]) for i in datasets_extended["アクセス"]]

fare_list = []
transfer_count_list = []
commute_time_list = []
for home_station in datasets_extended["nearest_st"]:
    trans_info = get_transfer_info(home_station, dest_station)
    transfer_count_list.append(trans_info["transfer_count"])
    fare_list.append(trans_info["fare"])
    commute_time_list.append(trans_info["ridetime"])

datasets_extended["commute_time"] = commute_time_list
datasets_extended["transfer_count"] = transfer_count_list
datasets_extended["fare"] = fare_list

## 通勤時間90分以内
datasets_extended = datasets_extended[datasets_extended["commute_time"] < 90]

# %% スコアリング ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import statistics as stat

# スコアリング対象の項目
tag_cols = [
    "val_monthly_fee",
    "val_area",
    "val_walk_time",
    "val_build_age",
    "diff"
    ]
inverter = [1, -1, 1, 1, -1]  # 1:低いほうが良い変数, -1:高いほうが良い変数
tag_weight = [0.3, 0.1, 0.2, 0.2, 0.1]

# 乗車時間があればスコアに含める
if include_ride_time:
    tag_cols.append("commute_time")
    inverter.append(1)
    tag_weight.append(0.1)

# 対象の項目ごとにスコアを算出
tag_cols_score = []
for tag_col, inv, w in zip(tag_cols, inverter, tag_weight):
    ave_ = stat.mean(datasets_extended[tag_col])
    std_ = stat.stdev(datasets_extended[tag_col])
    # 全体平均に対してどれだけ高いか低いか（差分）を評価する
    # 差分は標準偏差で割って正規化することで、Total Scoreを算出できるようにする
    datasets_extended[tag_col + "_score"] = [
        inv * w * (ave_ - val) / std_ for val in datasets_extended[tag_col]
    ]
    tag_cols_score.append(tag_col + "_score")

datasets_extended["Total_score"] = datasets_extended[tag_cols_score].sum(axis=1)
datasets_extended.sort_values("Total_score", ascending=False)
result = datasets_extended[
    ["間取り", "区", "nearest_st", "nearest_line", "transfer_count", "fare"]
    + tag_cols
    + tag_cols_score
    + ["Total_score", "名称", "URL"]
]
result.to_csv("result/scored_dataset.csv")
print('done')
