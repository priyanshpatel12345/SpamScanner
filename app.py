from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import os

app = Flask(__name__)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
mnb = pickle.load(open('model.pkl', 'rb'))

# Download necessary nltk data (fix typo 'punkt_tab' to 'punkt')
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transformation_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words("english") and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        email = request.form.get("email", "").strip()  # Get input and strip whitespace

        if email == "":
            prediction = "Please enter some email text before checking."
        else:
            cleaned = transformation_text(email)
            vector = tfidf.transform([cleaned])
            pred = mnb.predict(vector)[0]
            prediction = "Spam" if pred == 1 else "Not Spam"

    return render_template("index.html", Result=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render gives PORT as env variable
    app.run(host='0.0.0.0', port=port)