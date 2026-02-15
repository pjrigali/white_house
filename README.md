# White House News Scraper

A Python-based web scraper that collects news articles from [whitehouse.gov/news](https://www.whitehouse.gov/news/). Articles are saved as structured JSON files with metadata including title, date, category, content, and source URL.

## Features

- **Paginated Scraping** — Scrape across multiple pages of White House news listings.
- **Duplicate Detection** — Tracks previously collected articles via `article_tracking.csv` to avoid re-downloading.
- **Categorized Output** — Captures article categories (e.g., Articles, Briefings & Statements, Presidential Actions, Fact Sheets, Research).
- **Structured JSON** — Each article is saved as a clean JSON file with title, date, category, link, content, and scrape timestamp.

## Project Structure

```
white_house/
├── news_scraper.ipynb           # Main notebook to run the scraper
├── white_house_functions.py     # Core scraping logic and utilities
└── README.md
```

## Functions (`white_house_functions.py`)

| Function | Description |
|---|---|
| `slugify(text)` | Converts article titles into file-friendly slugs for JSON filenames. |
| `init_storage()` | Creates the output directory and initializes the tracking CSV with proper headers. |
| `is_already_collected(title)` | Checks the tracking CSV to determine if an article has already been scraped. |
| `update_tracking_csv(title, date_created, category)` | Appends a new entry to the tracking CSV with creation date, collection timestamp, title, and category. |
| `get_article_links(url)` | Scrapes a White House news listing page for article titles, links, dates, and categories. |
| `get_article_content(url)` | Fetches and extracts the full text content of an individual article page. |
| `save_article(article_data)` | Saves an article as a JSON file and updates the tracking CSV. |
| `scrape_news_pages(start_page, end_page)` | Orchestrates the full scraping pipeline across a range of listing pages. |

## How to Run

### Prerequisites

```bash
pip install requests beautifulsoup4
```

### Using the Notebook

1. Open `news_scraper.ipynb` in Jupyter or VS Code.
2. Set the page range in the first cell:
   ```python
   from white_house_functions import scrape_news_pages

   START_PAGE = 1
   END_PAGE = 5  # Adjust to scrape more pages (10 articles per page)
   ```
3. Run the second cell to execute the scraper:
   ```python
   new_articles, skipped = scrape_news_pages(START_PAGE, END_PAGE)
   ```

### Using Python Directly

```python
from white_house_functions import scrape_news_pages

new_articles, skipped = scrape_news_pages(start_page=1, end_page=10)
print(f"Collected {len(new_articles)} new articles, skipped {skipped} duplicates.")
```

## Output

Each article is saved as a JSON file with the following structure:

```json
{
    "title": "Article Title",
    "link": "https://www.whitehouse.gov/...",
    "date": "February 14, 2026",
    "category": "Articles",
    "content": "Full article text...",
    "scraped_at": "2026-02-14T23:45:00.000000"
}
```

An `article_tracking.csv` file tracks all collected articles:

| date_created | date_collected | article_name | category |
|---|---|---|---|
| February 14, 2026 | 2026-02-14 23:45:00 | Article Title | Articles |
