import json
import os
import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
from transformers import pipeline

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def analyze_articles(bronze_dir, silver_path):
    """
    Reads JSON articles from the bronze data lake directory,
    performs NLP analysis (Sentiment, Readability, Rhetoric, Political Leaning),
    and saves the aggregated results to a CSV in the silver data lake.
    """
    
    # Initialize analyzers
    sia = SentimentIntensityAnalyzer()
    
    print("Loading Zero-Shot Classification model (this may take a moment)...")
    # Using a distilled model for faster processing while maintaining reasonable accuracy
    classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
    rhetoric_labels = [
        "calm and neutral",
        "mildly assertive",
        "strongly opinionated",
        "aggressive and combative",
        "inflammatory and extreme"
    ]
    political_labels = ["progressive", "liberal", "moderate", "conservative", "far-right"]
    results = []
    
    if not os.path.exists(bronze_dir):
        print(f"Bronze directory not found: {bronze_dir}")
        return

    # Process each JSON file
    for filename in os.listdir(bronze_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(bronze_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    article = json.load(f)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
                
            content = article.get("content", "")
            title = article.get("title", "")
            date = article.get("date", "")
            category = article.get("category", "")
            
            if not content:
                print(f"Skipping {filename} - no content.")
                continue

            print(f"Analyzing: {title[:50]}...")
            
            # Clean content by removing title, date, and category boilerplate near the beginning
            clean_content = content.strip()
            for boilerplate in [category, title, date]:
                if boilerplate and boilerplate in clean_content[:300]:
                    clean_content = clean_content.replace(boilerplate, "", 1).strip()

            # 1. Sentiment Analysis (VADER)
            sentiment_scores = sia.polarity_scores(clean_content)
            compound_sentiment = sentiment_scores['compound']

            # 2. Grade Level / Readability (textstat)
            # textstat relies on syllable counts etc., which can be slow on huge texts. 
            # We'll calculate it on the entire text unless it proves too slow, then we can truncate.
            flesch_grade = textstat.flesch_kincaid_grade(clean_content)

            # 3. Zero-Shot Classification (Rhetoric & Political Spectrum)
            # Transformers have a token limit (typically 512 for standard models).
            # We will chunk the text into 400-word segments and average the scores.
            words = clean_content.split()
            chunk_size = 400
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
            
            # Prevent empty chunks
            if not chunks:
                chunks = [""]

            # Initialize lists to hold (score, weight) across all chunks
            rhet_results = {label: [] for label in rhetoric_labels}
            pol_results = {label: [] for label in political_labels}

            try:
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                        
                    weight = len(chunk.split())
                        
                    r_res = classifier(chunk, rhetoric_labels)
                    for lbl, score in zip(r_res['labels'], r_res['scores']):
                        rhet_results[lbl].append((score, weight))
                        
                    p_res = classifier(chunk, political_labels)
                    for lbl, score in zip(p_res['labels'], p_res['scores']):
                        pol_results[lbl].append((score, weight))

                if any(rhet_results.values()) and any(pol_results.values()):
                    # Weighted average the scores for each label across all chunks
                    
                    # Calculate total weight (all labels in rhet_results have the same weights collected)
                    total_weight = sum(w for _, w in next(iter(rhet_results.values()))) if rhet_results else 1
                    if total_weight == 0: total_weight = 1

                    avg_rhetoric = {lbl: sum(s * w for s, w in scores) / total_weight for lbl, scores in rhet_results.items() if scores}
                    avg_political = {lbl: sum(s * w for s, w in scores) / total_weight for lbl, scores in pol_results.items() if scores}
                    
                    # Find the label with the highest average score
                    top_rhetoric_label = max(avg_rhetoric, key=avg_rhetoric.get)
                    top_rhetoric_score = avg_rhetoric[top_rhetoric_label]
                    
                    top_political_label = max(avg_political, key=avg_political.get)
                    top_political_score = avg_political[top_political_label]
                else:
                    raise ValueError("No valid chunks processed.")
                
            except Exception as e:
                print(f"Zero-shot classification failed for {filename}: {e}")
                avg_rhetoric = {lbl: None for lbl in rhetoric_labels}
                avg_political = {lbl: None for lbl in political_labels}
                top_rhetoric_label = None
                top_rhetoric_score = None
                top_political_label = None
                top_political_score = None

            # Build the result row with all individual label scores
            row = {
                "title": title,
                "date": date,
                "category": category,
                "sentiment_compound": compound_sentiment,
                "positive_sentiment": sentiment_scores['pos'],
                "negative_sentiment": sentiment_scores['neg'],
                "flesch_kincaid_grade": flesch_grade,
                "top_rhetoric_label": top_rhetoric_label,
                "top_rhetoric_score": top_rhetoric_score,
                "top_political_label": top_political_label,
                "top_political_score": top_political_score,
                "source_file": filename
            }
            # Add individual rhetoric scores
            for lbl in rhetoric_labels:
                col_name = "rhetoric_" + lbl.replace(" ", "_")
                row[col_name] = avg_rhetoric.get(lbl)
            # Add individual political scores
            for lbl in political_labels:
                col_name = "political_" + lbl.replace(" ", "_")
                row[col_name] = avg_political.get(lbl)

            results.append(row)

    # Save to Silver Data Lake
    # Ensure silver directory exists
    silver_dir = os.path.dirname(silver_path)
    os.makedirs(silver_dir, exist_ok=True)
    
    if results:
        keys = results[0].keys()
        with open(silver_path, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
    else:
        # Create empty file if no results
        with open(silver_path, 'w', encoding='utf-8') as f:
            pass

    print(f"\nAnalysis complete. Results saved to {silver_path}")
    print(f"Processed {len(results)} articles.")

if __name__ == "__main__":
    bronze_dir = os.path.join(os.getcwd(), "data-lake", "01_Bronze", "white_house")
    silver_path = os.path.join(os.getcwd(), "data-lake", "02_Silver", "white_house", "article_analysis_results.csv")
    
    analyze_articles(bronze_dir, silver_path)
