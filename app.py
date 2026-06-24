import re
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template_string

model = joblib.load("book_rating_model.joblib")
lookups = joblib.load("book_rating_lookups.joblib")
author_count_map = lookups["author_book_count"]
publisher_count_map = lookups["publisher_book_count"]
author_ratings_map = lookups["author_mean_ratings"]

app = Flask(__name__)

languages = ["eng", "en-US", "en-GB", "spa", "fre", "ger", "jpn", "other"]

page = """
<!doctype html>
<html>
<head>
<title>Book Rating Predictor</title>
<style>
  body { font-family: sans-serif; background: whitesmoke; color: black; }
  .box { max-width: 460px; margin: 32px auto; background: white; padding: 24px;
         border: 1px solid lightgray; border-radius: 8px; }
  h2 { margin-top: 0; }
  label { display: block; margin-top: 12px; font-size: 14px; }
  input, select { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box;
                  border: 1px solid lightgray; border-radius: 4px; }
  button { margin-top: 18px; width: 100%; padding: 10px; border: none;
           border-radius: 4px; background: steelblue; color: white; font-size: 15px; cursor: pointer; }
  .hint { color: gray; font-size: 12px; }
  .result { margin-top: 20px; padding: 14px; background: honeydew;
            border: 1px solid lightgreen; border-radius: 4px; text-align: center; font-size: 18px; }
</style>
</head>
<body>
<div class="box">
<h2>Book Rating Predictor</h2>
<form method="post">
  <label>Title
    <input name="title" value="{{ title }}" required>
  </label>
  <label>Authors <span class="hint">use / between several authors</span>
    <input name="authors" value="{{ authors }}" required>
  </label>
  <label>Publisher
    <input name="publisher" value="{{ publisher }}" required>
  </label>
  <label>Number of pages
    <input name="num_pages" type="number" value="{{ num_pages }}">
  </label>
  <label>Number of ratings
    <input name="ratings_count" type="number" value="{{ ratings_count }}" required>
  </label>
  <label>Number of text reviews
    <input name="text_reviews_count" type="number" value="{{ text_reviews_count }}" required>
  </label>
  <label>Publication date <span class="hint">month/day/year</span>
    <input name="publication_date" value="{{ publication_date }}" required>
  </label>
  <label>Language
    <select name="language_code">
      <option value="eng" {{ "selected" if language_code == "eng" else "" }}>eng</option>
      <option value="en-US" {{ "selected" if language_code == "en-US" else "" }}>en-US</option>
      <option value="en-GB" {{ "selected" if language_code == "en-GB" else "" }}>en-GB</option>
      <option value="spa" {{ "selected" if language_code == "spa" else "" }}>spa</option>
      <option value="fre" {{ "selected" if language_code == "fre" else "" }}>fre</option>
      <option value="ger" {{ "selected" if language_code == "ger" else "" }}>ger</option>
      <option value="jpn" {{ "selected" if language_code == "jpn" else "" }}>jpn</option>
      <option value="other" {{ "selected" if language_code == "other" else "" }}>other</option>
    </select>
  </label>
  <button>Predict rating</button>
</form>
{% if prediction is not none %}
  <div class="result">Predicted rating: {{ prediction }} / 5</div>
{% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    title = "Harry Potter and the Half-Blood Prince"
    authors = "J.K. Rowling/Mary GrandPré"
    publisher = "Scholastic Inc."
    num_pages = "652"
    ratings_count = "2095690"
    text_reviews_count = "27591"
    publication_date = "9/16/2006"
    language_code = "eng"

    if request.method == "POST":
        title = request.form["title"].strip()
        authors = request.form["authors"].strip()
        publisher = request.form["publisher"].strip()
        num_pages = request.form["num_pages"].strip()
        ratings_count = request.form["ratings_count"].strip()
        text_reviews_count = request.form["text_reviews_count"].strip()
        publication_date = request.form["publication_date"].strip()
        language_code = request.form["language_code"].strip()

        ratings_value = float(ratings_count)
        reviews_value = float(text_reviews_count)
        first_author = authors.split("/")[0].strip()

        pub_date = pd.to_datetime(publication_date, format="%m/%d/%Y", errors="coerce")
        pub_year = pub_date.year if pd.notna(pub_date) else np.nan
        pub_quarter = pub_date.quarter if pd.notna(pub_date) else np.nan
        book_age = 2021 - pub_year

        num_pages_value = np.nan
        if num_pages != "" and float(num_pages) != 0:
            num_pages_value = float(num_pages)
        num_pages_missing = 1 if pd.isna(num_pages_value) else 0

        title_lower = title.lower()
        series_match = re.search(r"(\d+)\)\s*$", title)
        series_number = int(series_match.group(1)) if series_match else 0
        is_series = 1 if series_number > 0 else 0
        has_subtitle = 1 if ":" in title else 0
        has_number_in_title = 1 if re.search(r"\d", title) else 0
        is_collection = 1 if re.search(r"boxed set|box set|omnibus|collection|anthology", title_lower) else 0
        is_study_guide = 1 if re.search(r"cliffsnotes|sparknotes|spark notes|study guide", title_lower) else 0
        num_authors = authors.count("/") + 1
        title_word_count = len(title.split())
        title_char_len = len(title)

        log_ratings_count = np.log1p(ratings_value)
        log_text_reviews = np.log1p(reviews_value)
        reviews_per_rating = reviews_value / ratings_value
        if reviews_per_rating > 1:
            reviews_per_rating = 1
        if reviews_per_rating < 0:
            reviews_per_rating = 0

        if language_code.startswith("en"):
            lang_group = "english"
        elif language_code in ["spa", "fre", "ger", "jpn"]:
            lang_group = language_code
        else:
            lang_group = "other"

        publisher_norm = publisher.lower()
        publisher_norm = re.sub(r"[^a-z0-9 ]", " ", publisher_norm)
        publisher_norm = re.sub(r"\b(inc|ltd|llc|co|company|books|book|publishing|publishers|publications|press|group|the)\b", " ", publisher_norm)
        publisher_norm = re.sub(r"\s+", " ", publisher_norm).strip()

        author_book_count = author_count_map.get(first_author, 1)
        publisher_book_count = publisher_count_map.get(publisher_norm, 1)
        if first_author in author_ratings_map:
            author_avg_ratings = np.log1p(author_ratings_map[first_author])
        else:
            author_avg_ratings = np.log1p(ratings_value)

        n_editions = 1

        row = pd.DataFrame([{
            "num_pages": num_pages_value,
            "num_pages_missing": num_pages_missing,
            "num_authors": num_authors,
            "title_word_count": title_word_count,
            "title_char_len": title_char_len,
            "has_subtitle": has_subtitle,
            "has_number_in_title": has_number_in_title,
            "is_series": is_series,
            "series_number": series_number,
            "is_collection": is_collection,
            "is_study_guide": is_study_guide,
            "book_age": book_age,
            "pub_quarter": pub_quarter,
            "n_editions": n_editions,
            "author_book_count": author_book_count,
            "publisher_book_count": publisher_book_count,
            "author_avg_ratings": author_avg_ratings,
            "log_ratings_count": log_ratings_count,
            "log_text_reviews": log_text_reviews,
            "reviews_per_rating": reviews_per_rating,
            "first_author": first_author,
            "publisher_norm": publisher_norm,
            "lang_group": lang_group,
        }])

        value = model.predict(row)[0]
        value = max(0, min(5, value))
        prediction = round(value, 2)

    return render_template_string(page, prediction=prediction, title=title, authors=authors,
                                  publisher=publisher, num_pages=num_pages, ratings_count=ratings_count,
                                  text_reviews_count=text_reviews_count, publication_date=publication_date,
                                  language_code=language_code)

if __name__ == "__main__":
    app.run(debug=True)
