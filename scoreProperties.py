# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 01:56:59 2022

@author: chibi
"""
# %% ライブラリ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from retry import retry
import requests
import json
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
parser.add_argument("--monthly_fee", "-fee", help="Acceptable rent fee")
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
print(datasets.columns)

# 前処理：　学習モデル用に外れ値のデータ除去
# 希望の物件条件で絞るので削除
print('ダブり物件削除')
print('初期物件数：', len(datasets))
# datasets = datasets[datasets["monthly_fee"] < 30]
# print('家賃30万', len(datasets))

# datasets = datasets[datasets["walk_time"] < 20]
# print(datasets.shape)

# datasets = datasets[datasets["floor_area"] < 80]
# print(datasets.shape)

# datasets = datasets[datasets["build_age"] < 30]
# print(datasets.shape)

# ダブり物件削除
datasets = datasets.drop_duplicates(subset="URL")
datasets = datasets.drop_duplicates(subset="名称")
print('削除後の物件数：', len(datasets))

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
tag_col = ["monthly_fee"]
exp_cols = ["distance_to_station", "floor_area", "build_age", "floor_num"]

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
    print("Rental fee prediction model is already exist")
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
df_pred = pd.DataFrame({"monthly_fee": y_test, "monthly_fee_pred": y_pred})

# 散布図を描画(真値 vs 予測値)
plt.plot(
    y_test, y_test, color="red", label="x=y"
)  # 直線y = x (真値と予測値が同じ場合は直線状に点がプロットされる)
plt.scatter(y_pred, y_test)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.title("y vs y_pred")
plt.savefig("result/fee_prediction.png")

rmse = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE", rmse)
result = model.score(X_test, y_test)
print("R2", result)


# %% 予測値との差分を評価 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df_exp = datasets.loc[test_index, :].reset_index()
df_pred["market_gap"] = (
    df_pred["monthly_fee"] - df_pred["monthly_fee_pred"]
)
datasets_and_pred = pd.concat([df_exp, df_pred[["monthly_fee_pred", "market_gap"]]], axis=1)
datasets_and_pred["monthly_fee_pred"] = [round(val, 2) for val in datasets_and_pred["monthly_fee_pred"]]
datasets_and_pred["market_gap"] = [round(val, 2) for val in datasets_and_pred["market_gap"]]

print(datasets_and_pred.columns)

# %% 条件を好みに絞る ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 次に実行する勤務地までの乗車時間の算出に時間がかかるため、件数を絞ります。

## 家賃制限
if args.monthly_fee is not None:
    data_filtered = datasets_and_pred[datasets_and_pred["monthly_fee"] < float(args.monthly_fee)]
## 築年数の制限
if args.build_age is not None:
    data_filtered = data_filtered[data_filtered["build_age"] < float(args.build_age)]
## 間取りは2LDK, 1LDK, 2DK ('3LDK' '2LDK' '1K' '1LDK' 'ワンルーム' '1DK' '2DK' '2K' '3DK' '3SLDK' '1SLDK' '2SLDK' '4DK' '1SK' '4SLDK' より選択)
# data_filtered = data_filtered[
#     (data_filtered["間取り"] == "2LDK")
#     | (data_filtered["間取り"] == "1LDK")
#     | (data_filtered["間取り"] == "2DK")
# ]
## 床面積の制限
if args.floor_area is not None:
    data_filtered = data_filtered[data_filtered["floor_area"] > float(args.floor_area)]
## 駅からの距離（分）の制限
if args.walk_time is not None:
    data_filtered = data_filtered[data_filtered["distance_to_station"] < float(args.walk_time)]
## 足立区なんかには住まないので除外
data_filtered = data_filtered[data_filtered["区"] != '足立']

print('希望条件の物件数：', len(data_filtered))

# %% 物件からの通勤時間を導出 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# "get_transfer_info" is a private function wrritten in another file "yahoo_transfer.py"
from yahoo_transfer import get_transfer_info

print('Calculating ride time to the dest_station using Yahoo Transfers API.')
data_filtered["nearest_st"] = [
    (i.split("/")[1].split("駅")[0]) for i in data_filtered["アクセス"]
]
data_filtered["nearest_line"] = [(i.split("/")[0]) for i in data_filtered["アクセス"]]

fare_list = []
transfer_count_list = []
commute_time_list = []
for home_station in data_filtered["nearest_st"]:
    trans_info = get_transfer_info(home_station, dest_station)
    transfer_count_list.append(trans_info["transfer_count"])
    fare_list.append(trans_info["fare"])
    commute_time_list.append(trans_info["ridetime"])

data_filtered["commute_time"] = commute_time_list
data_filtered["transfer_count"] = transfer_count_list
data_filtered["fare"] = fare_list

## 通勤時間90分以内
data_filtered = data_filtered[data_filtered["commute_time"] < 90]

# %% スコアリング ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import statistics as stat

# スコアリング対象の項目
config_file_path = 'config.json'
with open(config_file_path, 'r') as f:
    config = json.load(f)

weights = config['weights']


target_cols = list(weights.keys())
target_weights = list(weights.values())
inverter = [1, -1, 1, 1, 1]  # 1:低いほうが良い変数, -1:高いほうが良い変数

# 乗車時間があればスコアに含める
if include_ride_time:
    target_cols.append("commute_time")
    inverter.append(1)
    target_weights.append(0.1)

# 対象の項目ごとにスコアを算出
target_cols_score = []
for tag_col, inv, w in zip(target_cols, inverter, target_weights):
    ave_ = stat.mean(data_filtered[tag_col])
    std_ = stat.stdev(data_filtered[tag_col])
    # 全体平均に対してどれだけ高いか低いか（差分）を評価する
    # 差分は標準偏差で割って正規化することで、Total Scoreを算出できるようにする
    data_filtered[tag_col + "_score"] = [
        round((inv * w * (ave_ - val) / std_), 3) for val in data_filtered[tag_col]
    ]
    target_cols_score.append(tag_col + "_score")

data_filtered["Total_score"] = data_filtered[target_cols_score].sum(axis=1)
data_filtered.sort_values("Total_score", ascending=False)
result = data_filtered[
    ["間取り", "区", "nearest_st", "nearest_line", "transfer_count", "fare"]
    + ["monthly_fee_pred"]
    + target_cols
    + target_cols_score
    + ["Total_score", "名称", "URL"]
]
result.to_csv("result/scored_dataset.csv")
print('done')
