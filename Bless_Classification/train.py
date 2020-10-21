# coding=utf-8

# @Author : Luojun
# @Email: luojun042@163.com
# @Time: 2020-09-23

import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import joblib
import time

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
STOP_WORDS_PATH = 'data/stop_words/baidu_stopwords.txt'
NEW_TRAIN_PATH = 'data/train_new.csv'
NEW_TEST_PATH = 'data/test_new.csv'
SAVE_MODEL = 'save_model/xgb.model'
SAVE_MODEL_TV = 'save_model/tv.pkl'

# 拼接训练集、测试集标签为List
def get_labels(train_data, test_data):

    train_labels = []
    test_labels = []

    for i in range(len(train_data["label"])):
        train_labels.append(train_data["label"][i])

    for i in range(len(test_data["label"])):
        test_labels.append(test_data["label"][i])

    return train_labels,test_labels

# 将删去停顿词的句子拼接成List
def delete_stopwords(sentences):

    sentence_list = []
    stopwords = {}.fromkeys([line.rstrip() for line in open(STOP_WORDS_PATH, encoding="utf-8")])
    stopwords = set(stopwords)

    for sentence in sentences:
        words = jieba.lcut(sentence)
        words = [w for w in words if w not in stopwords]
        new_sentence = " ".join(words)
        sentence_list.append(new_sentence)

    return sentence_list

# 将处理后的句子和标签构建为一个新的CSV
def after_clearn2csv(sentence_lists, labels,path):

    columns = ['contents','labels']

    save_file = pd.DataFrame(columns = columns, data = list(zip(sentence_lists, labels)))

    save_file.to_csv(path,index=False, encoding="utf-8")

if __name__ == "__main__":

    start_time = time.time()
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    train_labels, test_labels = get_labels(train_data, test_data)

    train_sentence_lists = delete_stopwords(train_data["bless_msg"])
    test_sentence_lists = delete_stopwords(test_data["bless_msg"])

    after_clearn2csv(train_sentence_lists,train_labels,NEW_TRAIN_PATH)
    after_clearn2csv(test_sentence_lists, test_labels,NEW_TEST_PATH)

    train_data = pd.read_csv(NEW_TRAIN_PATH, sep=',').astype(str)
    test_data = pd.read_csv(NEW_TEST_PATH, sep=',').astype(str)

    train_data["labels"] = train_data["labels"].apply(lambda x : int(x))
    test_data["labels"] = test_data["labels"].apply(lambda x: int(x))

    x_train = train_data["contents"]
    y_train = train_data["labels"]


    x_test = test_data["contents"]
    y_test = test_data["labels"]

    tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    tf_idf_train = tv.fit_transform(list(x_train))
    tf_idf_test = tv.transform(list(x_test))

    x_train_weight = tf_idf_train.toarray()
    x_test_weight = tf_idf_test.toarray()

    dtrain = xgb.DMatrix(x_train_weight, label=y_train)
    dtest = xgb.DMatrix(x_test_weight, label=y_test)

    params = {
        'booster':'gbtree',
        'silent': 0,
        'eta': 0.1,
        'max_depth': 6,
        'objective': 'multi:softprob',
        'num_class':12,
        'eval_metric': 'merror'
    }


    xgb_model = xgb.train(params, dtrain, num_boost_round = 100, evals = [(dtrain, 'train'), (dtest, 'test')])
    cost_time = time.time() - start_time
    print(f"Training finished!,cost {cost_time}(s)")
    xgb_model.save_model(SAVE_MODEL)
    joblib.dump(tv, SAVE_MODEL_TV)




























