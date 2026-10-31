"""
Simple web-review scraping template.

Use only websites that permit automated access and follow their
robots.txt and terms of service. For the college project, the
Kaggle dataset can be used as the primary data source.
"""

import requests
from bs4 import BeautifulSoup

def fetch_page(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.text

def extract_visible_text(html):
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(soup.stripped_strings)

if __name__ == "__main__":
    print("Scraper module ready. Use an allowed/public review page URL.")
