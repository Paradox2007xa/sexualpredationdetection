from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

###Loading model and cv
cv = pickle.load(open('cvx.pkl', 'rb'))
loaded_model = pickle.load(open('chatx.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        new_chat = request.form['chat']
        new_chat = re.sub('[^a-zA-Z]', ' ', new_chat)
        new_chat = new_chat.lower()
        new_chat = new_chat.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        new_chat = [ps.stem(word) for word in new_chat if not word in set(all_stopwords)]
        new_chat = ' '.join(new_chat)
        new_corpus = [new_chat]
        new_X_test = cv.transform(new_corpus).toarray()
        pred = loaded_model.predict(new_X_test)
        return render_template('result.html', prediction=pred)


if __name__ == "__main__":
    app.run(debug=True)
