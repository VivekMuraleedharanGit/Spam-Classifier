from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
tf= TfidfVectorizer(lowercase=True,ngram_range=(1,2),
                    stop_words = 'english',max_df = 0.5,min_df =2,use_idf =True)


app = Flask(__name__)
model = pickle.load(open('model', 'rb'))
tf= pickle.load(open('tf','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	

    '''
    For rendering results on HTML GUI
    '''
    if request.method=='POST':
    	message = request.form['message']
    	data= [message]
    	vect= tf.transform(data).toarray()
    	my_prediction=model.predict(vect)
    return render_template('result.html',prediction=my_prediction)

    


if __name__ == "__main__":
    app.run(debug=True)

