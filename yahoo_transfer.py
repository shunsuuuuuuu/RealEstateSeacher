import re
import urllib.request
from bs4 import BeautifulSoup
import urllib.parse # URLエンコード、デコード

# startsta = '東京' # 出発駅
# endsta = '横浜' # 到着駅

def get_transfer_info(startsta, endsta):
    startstaen = urllib.parse.quote(startsta) # encode
    endstaen = urllib.parse.quote(endsta) # encode

    url0 = 'https://transit.yahoo.co.jp/search/result?from='
    url1 = '&flatlon=&to='
    url2 = '&viacode=&viacode=&viacode=&shin=&ex=&hb=&al=&lb=&sr=&type=1&ws=3&s=&ei=&fl=1&tl=3&expkind=1&ticket=ic&mtf=1&userpass=0&detour_id=&fromgid=&togid=&kw='
    url = url0 + startstaen + url1 + endstaen + url2 + endstaen

    req = urllib.request.urlopen(url)
    html = req.read().decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')

    ridetime = soup.select_one('span.small').text.strip()
    ridetime = int(re.search(r'(\d+)分', ridetime).group(1))
    fare = soup.select_one('li.fare').text.strip()
    fare = int(re.search(r'(\d+)円', fare).group(1))
    transfer_count = soup.select_one('li.transfer').text.strip()
    transfer_count = int(re.search(r'乗換：(\d+)回', transfer_count).group(1))

    transfer_info = {"ridetime": ridetime, "fare": fare, "transfer_count": transfer_count}
    return transfer_info
