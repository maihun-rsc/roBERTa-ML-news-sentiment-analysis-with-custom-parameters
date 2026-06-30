import os
import json
import time
from typing import List, Dict
import logging

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from bs4 import BeautifulSoup
except ImportError:
    print("Please run: pip install -r requirements.txt")
    exit(1)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("selenium_scraper")

def setup_driver():
    options = Options()
    # Note: We do NOT run in headless mode by default. 
    # Cloudflare easily detects headless Chrome. Opening a real visible window bypasses 90% of WAFs.
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    
    # Mask webdriver to bypass bot-detection
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    # Execute CDP command to completely hide the webdriver flag from Javascript
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        """
    })
    
    return driver

def scrape_urls(urls: List[str], outlet_name: str, output_file: str):
    driver = setup_driver()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    success_count = 0
    with open(output_file, 'a', encoding='utf-8') as f:
        for url in urls:
            log.info(f"Scraping {url}")
            driver.set_page_load_timeout(15)
            try:
                driver.get(url)
            except Exception as e:
                log.warning(f"Page load timed out (or other error), but we will try to scrape what rendered so far: {e}")
            
            # Wait 5 seconds for Cloudflare challenges to clear and Javascript to load
            time.sleep(5) 
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            # Extract text from all paragraphs
            paragraphs = soup.find_all('p')
            text = " ".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            # Extract title
            title = soup.title.string if soup.title else "Unknown Title"
            log.info(f"Page title was: {title}")
            with open('debug.html', 'w', encoding='utf-8') as dbg:
                dbg.write(driver.page_source)
            
            if len(text) > 300:
                article_data = {
                    "url": url,
                    "title": title.strip(),
                    "text": text,
                    "outlet": outlet_name,
                    "timestamp": time.time()
                }
                f.write(json.dumps(article_data) + '\n')
                success_count += 1
                log.info(f"Success! Scraped {len(text)} characters.")
            else:
                log.warning(f"Failed: Extracted only {len(text)} characters. Cloudflare might still be blocking.")
                
            # Random sleep between requests to simulate human reading
            time.sleep(random.randint(3, 7))
            
    driver.quit()
    log.info(f"Finished scraping. Successfully extracted {success_count}/{len(urls)} articles.")

if __name__ == "__main__":
    import random
    
    # 1. Paste the URLs you want to scrape here
    target_urls = [
        "https://www.wionews.com/world"
    ]
    
    # 2. This is configured to output DIRECTLY into your main project's raw data folder!
    output_path = r"E:\news_sentiment\modules\data\raw\manual_selenium_scrape.jsonl"
    
    print("Testing WION scraper...")
    print(f"Data will be saved to: {output_path}")
    
    # 3. Uncomment the line below to actually run the scraper
    scrape_urls(target_urls, "WION", output_path)
