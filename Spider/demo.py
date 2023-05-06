
#https://www.bilibili.com/video/BV1jM4y1j7rb

import requests
import selenium
from lxml import etree

r = requests.get('https://gy.sustech.edu.cn/wxweb/#/bedSelect').text

print(r)