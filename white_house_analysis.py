import json
import os
import pandas as pd
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
    
    rhetoric_labels = ["moderate and measured", "strongly worded", "extreme rhetoric", "inflammatory"]
    political_labels = ["left wing", "center-left", "centrist", "center-right", "right wing"]

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

            # 1. Sentiment Analysis (VADER)
            sentiment_scores = sia.polarity_scores(content)
            compound_sentiment = sentiment_scores['compound']

            # 2. Grade Level / Readability (textstat)
            # textstat relies on syllable counts etc., which can be slow on huge texts. 
            # We'll calculate it on the entire text unless it proves too slow, then we can truncate.
            flesch_grade = textstat.flesch_kincaid_grade(content)

            # 3. Zero-Shot Classification (Rhetoric & Political Spectrum)
            # Transformers have a token limit (typically 512 for standard models).
            # We will truncate the text to the first ~400 words to represent the article's core message.
            words = content.split()
            content_truncated = " ".join(words[:400])
            
            try:
                rhetoric_result = classifier(content_truncated, rhetoric_labels)
                # get the top predicted label and its score
                top_rhetoric_label = rhetoric_result['labels'][0]
                top_rhetoric_score = rhetoric_result['scores'][0]
                
                # Check score for "extreme rhetoric" specifically
                extreme_idx = rhetoric_result['labels'].index("extreme rhetoric")
                extreme_score = rhetoric_result['scores'][extreme_idx]

                political_result = classifier(content_truncated, political_labels)
                top_political_label = political_result['labels'][0]
                top_political_score = political_result['scores'][0]

            except Exception as e:
                print(f"Zero-shot classification failed for {filename}: {e}")
                top_rhetoric_label = None
                top_rhetoric_score = None
                extreme_score = None
                top_political_label = None
                top_political_score = None

            # Append to results
            results.append({
                "title": title,
                "date": date,
                "category": category,
                "sentiment_compound": compound_sentiment,
                "positive_sentiment": sentiment_scores['pos'],
                "negative_sentiment": sentiment_scores['neg'],
                "flesch_kincaid_grade": flesch_grade,
                "top_rhetoric_label": top_rhetoric_label,
                "top_rhetoric_score": top_rhetoric_score,
                "extreme_rhetoric_score": extreme_score,
                "top_political_label": top_political_label,
                "top_political_score": top_political_score,
                "source_file": filename
            })

    # Save to Silver Data Lake
    df = pd.DataFrame(results)
    
    # Ensure silver directory exists
    silver_dir = os.path.dirname(silver_path)
    os.makedirs(silver_dir, exist_ok=True)
    
    df.to_csv(silver_path, index=False)
    print(f"\nAnalysis complete. Results saved to {silver_path}")
    print(f"Processed {len(results)} articles.")

if __name__ == "__main__":
    bronze_dir = os.path.join(os.getcwd(), ".data_lake", "01_Bronze", "white_house")
    silver_path = os.path.join(os.getcwd(), ".data_lake", "02_Silver", "white_house", "article_analysis_results.csv")
    
    analyze_articles(bronze_dir, silver_path)
