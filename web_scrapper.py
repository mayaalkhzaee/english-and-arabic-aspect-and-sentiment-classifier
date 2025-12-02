import os
import time
import json
import pickle
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

COOKIES_FILE = "amazon_cookies.pkl"
OUTPUT_FILE = "amazon_reviews_arabic.jsonl"

# 50 Arabic keywords
ARABIC_KEYWORDS = [
     "مكنسة كهربائية", "عطر", "مرتبة", "مروحة", "مكيف", "خلاط",
    "حفاضات", "كريم", "رضاعة", "سماعات", "شاشة", "لابتوب", "جوال",
    "بطارية متنقلة", "سماعة بلوتوث", "ميكرويف", "غسالة", "مجفف شعر",
    "مكواة", "قدور", "اواني طبخ", "شفاط", "فرن", "مكنسة لاسلكية",
    "مرتبة اطفال", "مرطب جو", "فلتر ماء", "عصارة", "طاولة", "كرسي",
    "خزانة", "سجاد", "ستارة", "بطانية", "مخدة", "وسادة", "مرتبة طبية",
    "حذاء", "شنطة", "سوار", "ساعة ذكية", "شاحن", "جراب جوال",
    "كشاف", "دريل", "عدة منزلية", "منظم اسلاك", "قلاية هوائية",
    "مقلاة", "ملعقة مطبخ"
]


##############################################
# Arabic detector
##############################################
def is_arabic(text):
    return re.search(r"[\u0600-\u06FF]", text) is not None


##############################################
# Browser launcher — clean session
##############################################
def launch_browser():
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.implicitly_wait(5)
    return driver


##############################################
# Manual login (one time only)
##############################################
def login_and_save_cookies():
    driver = launch_browser()
    print("🔐 Please log in manually to Amazon...")

    driver.get("https://www.amazon.ae/ap/signin")
    input("👉 After FULL login, press ENTER to save cookies...")

    cookies = driver.get_cookies()
    pickle.dump(cookies, open(COOKIES_FILE, "wb"))

    print("✔ Cookies saved.")
    driver.quit()


##############################################
# Load cookies
##############################################
def load_cookies(driver):
    cookies = pickle.load(open(COOKIES_FILE, "rb"))

    driver.get("https://www.amazon.ae/")
    time.sleep(2)

    for cookie in cookies:
        cookie.pop("sameSite", None)
        try:
            driver.add_cookie(cookie)
        except:
            pass

    driver.get("https://www.amazon.ae/")
    time.sleep(2)
    print("✔ Cookies loaded — logged in.")


##############################################
# Extract ASINs from a keyword
##############################################
def get_asins_from_keyword(keyword, max_items=10):
    if not os.path.exists(COOKIES_FILE):
        login_and_save_cookies()

    driver = launch_browser()
    load_cookies(driver)

    search_url = f"https://www.amazon.ae/s?k={keyword.replace(' ', '+')}"
    print("\n🔍 Searching keyword:", keyword)
    driver.get(search_url)
    time.sleep(2)

    blocks = driver.find_elements(By.CSS_SELECTOR, "div[data-component-type='s-search-result']")
    asins = []

    for block in blocks:
        asin = block.get_attribute("data-asin")
        if asin and len(asin) == 10:
            asins.append(asin)
            print("→ ASIN:", asin)
        if len(asins) >= max_items:
            break

    driver.quit()
    return asins


##############################################
# Append review line to file safely
##############################################
def append_review(review_obj):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(review_obj, ensure_ascii=False) + "\n")


##############################################
# Scrape all reviews (Arabic only)
##############################################
def get_all_reviews(asin, max_pages=50):
    driver = launch_browser()
    load_cookies(driver)

    url = f"https://www.amazon.ae/product-reviews/{asin}/?sortBy=recent&reviewerType=all_reviews"
    driver.get(url)
    time.sleep(2)

    page = 1

    while page <= max_pages:
        print(f"\n📄 Page {page} for ASIN {asin}")

        blocks = driver.find_elements(By.CSS_SELECTOR, "li[data-hook='review']")
        print("   → Reviews found:", len(blocks))

        if len(blocks) == 0:
            print("❌ No more reviews.")
            break

        for b in blocks:
            try:
                text = b.find_element(By.CSS_SELECTOR, "[data-hook='review-body']").text.strip()

                # Arabic filter
                if is_arabic(text):
                    review_obj = {
                        "asin": asin,
                        "page": page,
                        "text": text
                    }

                    append_review(review_obj)  # Save instantly
                    print("💾 Saved Arabic review.")

            except:
                pass

        # Try clicking next page
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, "li.a-last a")
            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(2)
            page += 1
        except:
            print("✔ Last page reached.")
            break

    driver.quit()


##############################################
# Run pipeline for all 10 Arabic keywords
##############################################
def run_pipeline():
    print("\n🌙 Starting Arabic Review Scraper...")
    print("Saving reviews to:", OUTPUT_FILE)

    # Clear file at start
    open(OUTPUT_FILE, "w", encoding="utf-8").close()

    for keyword in ARABIC_KEYWORDS:
        asins = get_asins_from_keyword(keyword)

        for asin in asins:
            print("\n==============================")
            print("📦 Scraping Arabic reviews for:", asin)
            print("==============================")

            get_all_reviews(asin)

    print("\n🎉 DONE — Arabic reviews saved to:", OUTPUT_FILE)


run_pipeline()
