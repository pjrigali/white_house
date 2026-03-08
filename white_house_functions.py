import requests
from bs4 import BeautifulSoup
import os
import json
import re
import csv
import random
from datetime import datetime

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
]

def get_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}

session = requests.Session()

OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.data_lake', '01_bronze', 'white_house')
TRACKING_FILE = os.path.join(OUTPUT_FOLDER, "article_tracking.csv")
COLLECTED_ARTICLES_CACHE = set()

def slugify(text):
    """Converts text into a file-friendly slug."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

def format_date_for_filename(date_str):
    """Converts a date string like 'January 27, 2026' to '20260127'."""
    try:
        # Expected format: "January 27, 2026"
        dt = datetime.strptime(date_str, "%B %d, %Y")
        return dt.strftime("%Y%m%d")
    except Exception:
        # Try a few other common formats if needed
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d %B %Y"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y%m%d")
            except:
                continue
        return "00000000"

def init_storage():
    """Initializes the output folder and tracking CSV if they don't exist."""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # Check if tracking file exists and has the correct header
    headers = ['date_created', 'date_collected', 'article_name', 'category', 'filename']
    file_exists = os.path.exists(TRACKING_FILE)
    
    if file_exists:
        with open(TRACKING_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            first_line = next(reader, None)
            if first_line != headers:
                # Need to update header
                print("Updating tracking CSV header to include 'category'...")
                rows = list(reader)
                with open(TRACKING_FILE, 'w', newline='', encoding='utf-8') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(headers)
                    for row in rows:
                        if len(row) == 3:
                            row.append("Unknown") # Category
                        if len(row) == 4:
                            # Try to reconstruct filename if missing (for migration)
                            row.append("None") 
                        writer.writerow(row)
    else:
        with open(TRACKING_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
    # Load tracking data into memory cache
    COLLECTED_ARTICLES_CACHE.clear()
    with open(TRACKING_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            COLLECTED_ARTICLES_CACHE.add((row['article_name'], row['date_created']))

def is_already_collected(title, date):
    """Checks if an article with the given title and date has already been collected."""
    return (title, date) in COLLECTED_ARTICLES_CACHE

def update_tracking_csv(title, date_created, category, filename):
    """Adds a new entry to the tracking CSV."""
    date_collected = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TRACKING_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([date_created, date_collected, title, category, filename])
    # Also update cache
    COLLECTED_ARTICLES_CACHE.add((title, date_created))

def get_article_links(url):
    """Scrapes article titles, links, dates, and categories from a list page."""
    import time
    max_retries = 5
    for attempt in range(max_retries):
        response = session.get(url, headers=get_headers())
        if response.status_code == 200:
            break
        elif response.status_code in [403, 429]:
            wait_time = random.uniform(10.0, 20.0) * (attempt + 1)
            print(f"Rate limited ({response.status_code}) on {url}. Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        else:
            print(f"Failed to fetch {url}: {response.status_code}")
            return []
    else:
        print(f"Failed to fetch {url} after {max_retries} attempts.")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.select('li.wp-block-post')
    
    data = []
    for article in articles:
        title_tag = article.select_one('.wp-block-post-title a')
        if not title_tag:
            continue
            
        title = title_tag.get_text(strip=True)
        link = title_tag['href']
        
        date_text = "N/A"
        category = "Uncategorized"
        
        meta = article.select_one('.wp-block-whitehouse-post-template__meta')
        if meta:
            # Extract date
            time_tag = meta.find('time')
            if time_tag:
                date_text = time_tag.get_text(strip=True)
            else:
                # Try to find date in meta text if no time tag
                date_text = meta.get_text(strip=True)
            
            # Extract category
            cat_tag = meta.select_one('.taxonomy-category a')
            if cat_tag:
                category = cat_tag.get_text(strip=True)
        
        data.append({
            'title': title,
            'link': link,
            'date': date_text,
            'category': category
        })
    
    return data

def get_article_content(url):
    """Scrapes the main text content of an individual article."""
    import time
    max_retries = 5
    for attempt in range(max_retries):
        response = session.get(url, headers=get_headers())
        if response.status_code == 200:
            break
        elif response.status_code == 404:
            return ""  # Page not found, no point retrying
        elif response.status_code in [403, 429]:
            wait_time = random.uniform(10.0, 20.0) * (attempt + 1)
            print(f"  * Rate limited fetching content on {url}. Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        else:
            return ""
    else:
        print(f"  * Failed to fetch content for {url} after retries.")
        return ""
    
    soup = BeautifulSoup(response.text, 'html.parser')
    content_div = soup.select_one('.wp-block-post-content')
    if not content_div:
        content_div = soup.select_one('article')
    
    if content_div:
        return content_div.get_text(separator='\n', strip=True)
    return ""

def save_article(article_data):
    """Saves article data to a JSON file and updates tracking CSV."""
    date_prefix = format_date_for_filename(article_data['date'])
    name_slug = slugify(article_data['title'][:100])
    filename = f"{date_prefix}_{name_slug}.json"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    
    # Add scrape timestamp
    article_data['scraped_at'] = datetime.now().isoformat()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(article_data, f, indent=4, ensure_ascii=False)
    
    # Update tracking CSV
    update_tracking_csv(article_data['title'], article_data['date'], article_data['category'], filename)
    
    return filepath

def scrape_news_pages(start_page=1, end_page=1):
    """Orchestrates scraping across a range of pages."""
    init_storage()
    all_new_articles = []
    total_skipped = 0
    import time
    
    for page_num in range(start_page, end_page + 1):
        if page_num == 1:
            url = "https://www.whitehouse.gov/news/"
        else:
            url = f"https://www.whitehouse.gov/news/page/{page_num}/"
        
        print(f"\n--- Scraping Page {page_num}: {url} ---")
        articles = get_article_links(url)
        
        if not articles:
            print(f"No articles found on page {page_num}. Stopping.")
            break
            
        print(f"Found {len(articles)} articles candidate.")
        
        for article in articles:
            title = article['title']
            date = article['date']
            if is_already_collected(title, date):
                print(f"  - Skipping duplicate: {title} ({date})")
                total_skipped += 1
                continue
            
            print(f"  - Fetching [{article['category']}]: {title}")
            content = get_article_content(article['link'])
            article['content'] = content
            
            save_article(article)
            all_new_articles.append(article)
            # Polite delay after fetching article
            time.sleep(random.uniform(2.0, 4.0))
            
        # Polite delay after each page
        time.sleep(random.uniform(1.5, 3.0))
            
    return all_new_articles, total_skipped
