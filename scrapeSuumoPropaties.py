# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 01:51:26 2022

@author: chibi
"""

#%%
from retry import retry
import requests
from bs4 import BeautifulSoup
import pandas as pd 
import matplotlib.pyplot as plt
import argparse

@retry(tries=3, delay=10, backoff=2)
def get_html(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    return soup

parser = argparse.ArgumentParser(description="Command line argument parser")
parser.add_argument("--url", "-u", help="URL of Sumo's property listing page")
parser.add_argument("--page_num", "-pg", help="Number of pages")
args = parser.parse_args()

# 引数がない場合は東京23区で実行
if args.url is not None:
    base_url = args.url
else:
    base_url = "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&ta=13&sc=13101&sc=13102&sc=13103&sc=13104&sc=13105&sc=13113&sc=13106&sc=13107&sc=13108&sc=13118&sc=13121&sc=13122&sc=13123&sc=13109&sc=13110&sc=13111&sc=13112&sc=13114&sc=13115&sc=13120&sc=13116&sc=13117&sc=13119&cb=0.0&ct=9999999&et=9999999&cn=9999999&mb=0&mt=9999999&shkr1=03&shkr2=03&shkr3=03&shkr4=03&fw2=&srch_navi=1"

soup = get_html(base_url)
if args.page_num is not None:
    max_page = int(args.page_num)
else:
    max_page = int(soup.select_one('ol.pagination-parts').text.strip().split()[-1])

#%%
all_data = []
for page in range(1, max_page+1):
    print("page", page, "items", max_page)
    # url per page 
    url = base_url + '&page={page}'.format(page=page)
    
    # get html
    soup = get_html(url)
    
    # extract all items
    items = soup.findAll("div", {"class": "cassetteitem"})
    
    # process each item
    for item in items:
        stations = item.findAll("div", {"class": "cassetteitem_detail-text"})
        # process each station 
        for station in stations:
            # define variable 
            base_data = {}

            # collect base information    
            base_data["名称"] = item.find("div", {"class": "cassetteitem_content-title"}).getText().strip()
            base_data["カテゴリ"] = item.find("div", {"class": "cassetteitem_content-label"}).getText().strip()
            base_data["アドレス"] = item.find("li", {"class": "cassetteitem_detail-col1"}).getText().strip()
            base_data["アクセス"] = station.getText().strip()
            base_data["築年数"] = item.find("li", {"class": "cassetteitem_detail-col3"}).findAll("div")[0].getText().strip()
            base_data["構造"] = item.find("li", {"class": "cassetteitem_detail-col3"}).findAll("div")[1].getText().strip()
            
            # get each room's propaty 
            tbodys = item.find("table", {"class": "cassetteitem_other"}).findAll("tbody")
            
            for tbody in tbodys:
                data = base_data.copy()

                data["階数"] = tbody.findAll("td")[2].getText().strip()

                data["家賃"] = tbody.findAll("td")[3].findAll("li")[0].getText().strip()
                data["管理費"] = tbody.findAll("td")[3].findAll("li")[1].getText().strip()

                data["敷金"] = tbody.findAll("td")[4].findAll("li")[0].getText().strip()
                data["礼金"] = tbody.findAll("td")[4].findAll("li")[1].getText().strip()

                data["間取り"] = tbody.findAll("td")[5].findAll("li")[0].getText().strip()
                data["面積"] = tbody.findAll("td")[5].findAll("li")[1].getText().strip()
                
                data["URL"] = "https://suumo.jp" + tbody.findAll("td")[8].find("a").get("href")
                
                all_data.append(data)    

    df = pd.DataFrame(all_data)
    # df.to_csv('database.csv')

#%% 数値変換
import re
df_numeric = df.copy()
# バスを必要とする物件を除外して、最寄り駅までの徒歩時間を計算
isWalkable = df['アクセス'].str.contains('歩')
df_numeric=df_numeric.loc[isWalkable]
useBus = df['アクセス'].str.contains('バス')
df_numeric=df_numeric.loc[~useBus]
df_numeric["val_walk_time"] = [int(i.split('歩')[1].split('分')[0]) for i in df_numeric['アクセス']]

# 築年数を数値に変換
df_numeric['築年数'] = [i.replace('新築','築0年') for i in df_numeric['築年数']]
df_numeric['val_build_age'] = [int(re.sub(r"\D", "", i)) for i in df_numeric['築年数']]

# 建物の階数を数値に変換
## ハイフンのみが表記されている行を削除
df_numeric = df_numeric[~df_numeric['階数'].str.contains('-$')]
floor_num = df_numeric['階数'].str.extract(r'(\d+)[^\d]*$').astype(float)
df_numeric['val_floor'] = floor_num

# 家賃を数値に変換
df_numeric['val_rental_fee'] = [float(i.split('万円')[0]) for i in df_numeric['家賃']]

# 管理費を数値に変換
df_numeric['管理費'] = [i.replace('-','0') for i in df_numeric['管理費']]
df_numeric['val_service_fee'] = [int(i.split('円')[0])/10000 for i in df_numeric['管理費']]

# 家賃+管理費を計算
df_numeric['val_monthly_fee'] = df_numeric['val_rental_fee'] + df_numeric['val_service_fee']

# 敷金と礼金を数値に変換
df_numeric['敷金'] = [i.replace('-','0') for i in df_numeric['敷金']]
df_numeric['val_deposit'] = [float(i.split('万円')[0]) for i in df_numeric['敷金']]
df_numeric['礼金'] = [i.replace('-','0') for i in df_numeric['礼金']]
df_numeric['val_Reward'] = [float(i.split('万円')[0]) for i in df_numeric['礼金']]

# 部屋面積を数値に変換
df_numeric['val_area'] = [float(i.split('m2')[0]) for i in df_numeric['面積']]

# 住所から区を抽出
# df_numeric['section'] = [(i.split('区')[0]).replace('東京都', '') for i in df_numeric['アドレス']]

# 保存
df_numeric.to_csv('result/datasets.csv')

#%% 相関を見る
# df_numeric = df_numeric[df_numeric['val_monthly_fee']<100]
plt.scatter(df_numeric['val_area'],df_numeric['val_monthly_fee'])
plt.xlabel('Area_size', fontsize=12)
plt.ylabel('Monthly fee', fontsize=12)
plt.show()
