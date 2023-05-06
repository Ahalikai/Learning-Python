import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


options = webdriver.ChromeOptions()
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36")
# 这个是绝对路径

cookies = [{"name": "_ga_0KD226TRZ5", "value": "GS1.1.1692270661.1.1.1692271979.0.0.0"},
            {"name": "_ga", "value": "GA1.1.288898560.1691587142"},
            {"name": "_ga_LE73XPNH8J", "value": "GS1.3.1691587142.1.1.1691587411.0.0.0"}
           ]

driver = webdriver.Chrome(options=options)

# for cookie in cookies:
#     driver.add_cookie(cookie)

driver.get("https://gy.sustech.edu.cn/wxweb/#/bedSelect")

time.sleep(40)

# driver.find_element(By.CLASS_NAME, 'van-sidebar-item__text').click()
driver.find_element(By.LINK_TEXT, "二期").click()
time.sleep(2)
driver.find_element(By.LINK_TEXT, "二期10栋").click()
time.sleep(2)
driver.find_element(By.LINK_TEXT, "3层").click()
time.sleep(2)
driver.switch_to.window(driver.window_handles[-1])
driver.switch_to.default_content()
time.sleep(2)
print("XPath")
button1 = driver.find_element(By.CLASS_NAME, "van-row")
print("XPath")
driver = button1.find_element(By.CLASS_NAME, "van-col van-col--20")
print("XPath")
driver.find_element(By.LINK_TEXT, "双人间,无床垫").click()
time.sleep(2)
driver.find_element(By.LINK_TEXT, "1号床").click()
time.sleep(2)
driver.find_element(By.LINK_TEXT, "收藏").click()
# # 最大化浏览器
# driver.maximize_window()
# time.sleep(3)
#driver.close()