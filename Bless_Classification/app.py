# coding=utf-8

from flask import Flask
import flask
from train import delete_stopwords
from train import SAVE_MODEL_TV
from train import SAVE_MODEL
import xgboost as xgb
import joblib
from flask import request

app = Flask(__name__)

def sentence_prediction(sentence):

    sentence = delete_stopwords(sentence)

    tv = joblib.load(SAVE_MODEL_TV)
    tf_idf = tv.transform(sentence)
    predict_weight = tf_idf.toarray();

    xgb_predict = xgb.Booster(model_file=SAVE_MODEL)
    dpredict = xgb.DMatrix(predict_weight)

    predict_prob = xgb_predict.predict(dpredict)

    return predict_prob


@app.route("/predict/",methods = ["GET"])
def predict():

    sentence = [request.args.get("sentence")]
    predict_prob = sentence_prediction(sentence)
    response = {
        'birthday': predict_prob[0][0].item(),
        'classmate': predict_prob[0][1].item(),
        'business': predict_prob[0][2].item(),
        'love': predict_prob[0][3].item(),
        'friend': predict_prob[0][4].item(),
        'colleague': predict_prob[0][5].item(),
        'family': predict_prob[0][6].item(),
        'single': predict_prob[0][7].item(),
        'old': predict_prob[0][8].item(),
        'wedding': predict_prob[0][9].item(),
        'celebrate': predict_prob[0][10].item(),
        'other': predict_prob[0][11].item()
    }

    return flask.jsonify(response)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)