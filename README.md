# 概要
SUUMOのWebページからスクレイピングで不動産情報を取得し、構造化データに変換するツールです。

# 環境構築
```bash
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

# 実行方法
## 手順1：スクレイピングで物件データを取得
スーモの物件一覧ページから物件データをスクレイピングし、家賃や築年数などの情報を数値データに変換します。
```
python scrapeSuumoPropaties.py -u "{URL}" -pg {page_num}
```
{URL}にはスーモの物件一覧ページのURLを、{page_num}には取得するページ数を入力してください。  
例：[東京23区の物件一覧](https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&ta=13&sc=13101&sc=13102&sc=13103&sc=13104&sc=13105&sc=13113&sc=13106&sc=13107&sc=13108&sc=13118&sc=13121&sc=13122&sc=13123&sc=13109&sc=13110&sc=13111&sc=13112&sc=13114&sc=13115&sc=13120&sc=13116&sc=13117&sc=13119&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shkr1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&pc=50)  

引数がない場合、東京23区、全ページで実行されますが、データ取得に1時間ほどかかりますので注意してください。予め条件で絞った方がいいかもしれません。  
取得したデータは result/datasets.csv で出力されます。  

実行例↓
```
python scrapeSuumoPropaties.py -u "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&pc=50&smk=&po1=25&po2=99&shkr1=03&shkr2=03&shkr3=03&shkr4=03&sc=13101&sc=13102&sc=13103&sc=13104&sc=13105&sc=13113&sc=13106&sc=13107&sc=13108&sc=13118&sc=13121&sc=13122&sc=13123&sc=13109&sc=13110&sc=13111&sc=13112&sc=13114&sc=13115&sc=13120&sc=13116&sc=13117&sc=13119&ta=13&cb=0.0&ct=20.0&md=03&md=04&md=05&md=06&md=07&md=08&md=09&md=10&md=11&md=12&md=13&md=14&et=20&mb=0&mt=9999999&cn=20&fw2="
```

## 手順2：物件データをスコアリング
物件ごとにスコア付けを行います。割安感を評価するため、機械学習で予測した値に対する差分を算出します。また通勤時間を評価するため、[Yahoo乗り換え](https://transit.yahoo.co.jp/)をスクレイピングして物件の最寄り駅から指定した駅までの乗車時間を取得します。

Linux環境の場合、実行前に `libgomp1` をインストールする必要があります。
```
 sudo apt update
 sudo apt install libgomp1
```
実行コマンド
```
python scoreProperties.py -ds {dest_station} -fee {rent_fee} -age {build_age} -area {area} -t {walk_time}
```
{dest_station} には[Yahoo乗り換え](https://transit.yahoo.co.jp/)に入力する勤務地までの駅名を入力してください。指定しない場合、東京駅を目的地とした乗車時間が算出され、スコア算出に通勤時間は含まれません。  
通勤時間の算出には処理時間がかかります。物件数をある程度絞るため、最低限の条件を引数に設定して下さい。指定しない場合、物件数の絞り込みは行われません。
* --fee [int]: 許容できる家賃（万円）
* --age [int]: 許容できる築年数（年）
* --area [int]: 最低限の床面積 （m2）
* --t [int]: 許容できる最寄駅までの徒歩時間（分）
  
実行例↓
```
python scoreProperties.py -ds 大崎 -fee 18 -age 10 -area 40 -t 15
```

実行完了すると、result/scored_dataset.csv が出力されます。スコアが高い物件ほど、あなたの希望にマッチした物件です。

## 手順3：スコアの分析と可視化
スコアが高かった物件の情報を棒グラフで表示します。その他、区ごとの傾向分析や築年数による家賃の変化などを分析しています。ご自身で追加で分析してみてください。
```
python analizeScoreResults.py
```

