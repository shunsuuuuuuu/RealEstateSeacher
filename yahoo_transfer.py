import re
import urllib.request
from bs4 import BeautifulSoup
import urllib.parse # URLエンコード、デコード

# startsta = '東京' # 出発駅
# endsta = '横浜' # 到着駅

def get_transfer_info(startsta, endsta):
    startstaen = urllib.parse.quote(startsta) # encode
    endstaen = urllib.parse.quote(endsta) # encode

    # 電車の利用時間
    hh, m1, m2 = "09", "0", "0"

    url0 = 'https://transit.yahoo.co.jp/search/result?from='
    url1 = '&flatlon=&to='
    url2 = f'&viacode=&viacode=&viacode=&y=2024&m=12&d=25&hh={hh}&m1={m1}&m2={m2}&shin=&ex=&hb=&al=&lb=&sr=&type=1&ws=3&s=&ei=&fl=1&tl=3&expkind=1&ticket=ic&mtf=1&userpass=0&detour_id=&fromgid=&togid=&kw='
    url = url0 + startstaen + url1 + endstaen + url2 + endstaen

    req = urllib.request.urlopen(url)
    html = req.read().decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')

    ridetime_text = soup.select_one('span.small').text.strip()
    # 正規表現で時間を抽出
    ridetime_hour_match = re.search(r'(\d+)時間', ridetime_text)
    ridetime_min_match = re.search(r'(\d+)分', ridetime_text)
    # 時間がない場合は０で定義
    ridetime_hour = int(ridetime_hour_match.group(1)) if ridetime_hour_match else 0
    ridetime_min = int(ridetime_min_match.group(1)) if ridetime_min_match else 0
    ridetime = ridetime_hour * 60 + ridetime_min
    
    fare = soup.select_one('li.fare').text.strip()
    fare = int(re.search(r'(\d+)円', fare).group(1))
    transfer_count = soup.select_one('li.transfer').text.strip()
    transfer_count = int(re.search(r'乗換：(\d+)回', transfer_count).group(1))

    transfer_info = {"ridetime": ridetime, "fare": fare, "transfer_count": transfer_count}
    return transfer_info
