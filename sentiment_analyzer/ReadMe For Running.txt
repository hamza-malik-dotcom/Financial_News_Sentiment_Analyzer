# Financial News Sentiment Analyzer

This is a Streamlit-based web app that classifies financial news headlines as **Positive**, **Negative**, or **Neutral** using multiple NLP models, including:

- Logistic Regression (TF-IDF)
- LSTM
- FinBERT (Transformers)
- Word2Vec + LSTM

---

## 🔧 Requirements

Install dependencies in a Python 3.10.10 virtual environment:

```bash
python -m venv venv
# Activate virtual environment:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies:
pip install -r requirements.txt (Requirement file is present inside "sentiment_analyzer")



-------------------------------
How To Run It
-------------------------------

Step 1 : Open the folder "sentiment_analyzer" in VsCode.
Step 2 : Create the python environment with the following commands "python -m venv venv" -----> "venv\Scripts\activate".
Step 3 : Once the envionment is created run "streamlit run app.py" it will take you to web with running server "http://localhost:8501/" where you can test the models.
Step 4 : Some of the predictions are listed below for your ease:

-----------------------
Testing News 
----------------------

Few example headlines from the Financial PhraseBank dataset you can use to test your app:

🔹 Positive:
"The company reported better than expected quarterly earnings."

"Shares rose after the firm announced a major partnership."

"The outlook for the next quarter is promising."

🔹 Negative:
"The firm posted a significant loss in the last fiscal year."

"The CEO resigned amid falling profits and internal conflict."

"Sales have declined sharply due to weak consumer demand."

🔹 Neutral:
"The board of directors held a meeting on Monday."

"The company operates in over 30 countries worldwide."

"The stock was unchanged during the trading session."
