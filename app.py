from flask import Flask, render_template, request
import pickle
import string
import numpy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
model = pickle.load(open("main_proj.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def basic():
    return render_template('index.html')


@app.route('/graphs', methods=['GET', 'POST'])
def graphs():
    return render_template('graphs.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/team', methods=['GET', 'POST'])
def team():
    return render_template('team.html')

@app.route('/twitter', methods=['GET', 'POST'])
def twitter():
    return render_template('twitter_analysis.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('prediction.html')

@app.route('/sentiment_analysis', methods=['GET', 'POST'])
def sentiment_analysis():
    return render_template('sentiment_analysis.html')




@app.route('/sentiment_prediction', methods=['GET', 'POST'])
def sentiment_prediction():
    if request.method == 'POST':
        txt = request.form['sentiment_input']
        words = txt.split()
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in words]
        str1 = " "  
        clean_text = (str1.join(stripped))
        analysis = SentimentIntensityAnalyzer()
        score = analysis.polarity_scores(clean_text)
        del score['compound']
        max_value = max(score,key=score.get)
        positive = 'The given text has a positive precedence'
        negative = 'The given text has a negative precedence'
        neutral = 'The given text shows no polarity, might be neutral'
        if max_value=='pos':
            return render_template('sentiment_prediction.html', positive=positive)
       # elif analysis.sentiment.polarity==0 :
           # return render_template('sentiment_prediction.html', neutral=neutral)
        elif max_value=='neg':
            return render_template('sentiment_prediction.html', negative=negative)
        else:
            return render_template('sentiment_prediction.html', neutral=neutral)
    return render_template('sentiment_prediction.html')



@app.route('/result_prediction', methods=['GET', 'POST'])
def result_prediction():
    if request.method == 'POST':
        GENERAL_VOTES = request.form['GENERAL_VOTES']
        POSTAL_VOTES = request.form['POSTAL_VOTES']
        TOTAL_VOTES = request.form['TOTAL_VOTES']
        OVER_TOTAL_ELECTORS_IN_CONSTITUENCY = request.form['OVER_TOTAL_ELECTORS_IN_CONSTITUENCY']
        OVER_TOTAL_VOTES_POLLED_IN_CONSTITUENCY = request.form['OVER_TOTAL_VOTES_POLLED_IN_CONSTITUENCY']
        y_pred = [[GENERAL_VOTES, POSTAL_VOTES, TOTAL_VOTES, OVER_TOTAL_ELECTORS_IN_CONSTITUENCY,OVER_TOTAL_VOTES_POLLED_IN_CONSTITUENCY]]
        
        prediction_value = model.predict(y_pred)
        lose = 'Lose'
        win = 'Winner'
        
        if prediction_value == 0:
            return render_template('result_prediction.html', lose=lose)
        else:
            return render_template('result_prediction.html', win=win)
    return render_template('result_prediction.html')
if __name__ == '__main__':
    app.run(debug=True)
