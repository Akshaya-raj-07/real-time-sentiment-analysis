# Real-Time Sentiment Analysis with Web Scraping

A 3rd-year AIML project for analyzing product reviews using NLP and Machine Learning.

## Main pipeline
1. Load review data
2. Data quality checks
3. Remove empty/duplicate records
4. Clean review text
5. Create Positive / Neutral / Negative labels from ratings
6. TF-IDF feature extraction
7. Train and compare ML models
8. Ensemble prediction
9. Product comparison
10. Pain-point detection
11. Negative sentiment alerts
12. Streamlit dashboard

## Dataset
Place the Kaggle CSV file here:
`data/raw/amazon_vfl_reviews.csv`

The code expects these columns:
`asin`, `name`, `date`, `rating`, `review`

## Run
Activate the virtual environment, install requirements, then:

```bash
python src/preprocessing.py
python src/features.py
python src/train.py
python src/comparison.py
streamlit run dashboard/app.py
```

The trained model is saved in `models/`.
The processed dataset is saved in `data/processed/`.

## Note
The original Kaggle dataset is not included in this ZIP because it is an external dataset. Put your downloaded CSV in `data/raw/`.
