from flask import Flask, request, render_template
import pickle
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

CORS(app)
#chargement des modèles œ&
file1 = open("count_vectorizer_model.sav", 'rb')
vectorizer = pickle.load(file1)

file2 = open("logistic_regression_model.sav", 'rb')
lr = pickle.load(file2)

#@app.route("/index.html", methods=["GET"])
#def vue():
#    return render_template('/index.html')

@app.route("/api/spamdetector", methods=["GET"])
def detector():
    mail = request.args.get("mail")
  
    mail_2 = vectorizer.transform([mail]).toarray()
    p = lr.predict_proba(mail_2.reshape(1, -1))[0]

    print("Ce mail est un ham à ",p[1],"%")
    print("Ce mail est un spam à",p[0],"%")
    #message=""
    #if p[0]>=p[1]:
    message = "ce mail est un spam à " + str(p[0]) + " %."
    #else :
    #    message = "Ce mail est un ham"
    
    return message



if __name__ == "__main__":
    app.run(port=8001, debug=True, host="localhost")
    