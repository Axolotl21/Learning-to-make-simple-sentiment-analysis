import io, os, csv, ast, re, string
from PIL.Image import Image
from flask import Flask, render_template, request
from matplotlib import image
from nltk.corpus.reader.chasen import test
import tweepy, csv
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
import wordcloud as wc
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

API_KEY = "N7O3baZqiurPZs8DDS1WUmEdr"
API_SECRET_KEY = "2lixxTS2zLhSpf3RNyaR9mKs8dZc2g4nT2d5urImpl97W019ol"
ACCESS_TOKEN = "1345318319490928640-gIWe7GczilwOlwUXw8Gmxf97H3CcZt" 
ACCESS_TOKEN_SECRET= "hcldRkr82yAV9L4T4vRFAizqP221szfnTMxv4A14Bc9Yt"

auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth) 

loaded_model = pickle.load(open("Pickle_NBC_Model.pkl", 'rb'))
loaded_vec = pickle.load(open("Pickle_Vectorizer.pkl", 'rb'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

processing_filepath = "static\\file\\lexicon\\processing.csv"
lexicon_filepath = "static\\file\\lexicon\\labeling.csv"

def open_file(filepath):
    data = []
    with open(filepath, encoding="utf8", errors='ignore') as file:
        csv_file = csv.reader(file)
        for row in csv_file:
                data.append(row)
    return data

def case_folding(data):
    data = data.lower()
    return data

def cleansing(data):
    data = data.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    data = data.encode('ascii', 'replace').decode('ascii')
    data = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", data).split())
    data = re.sub(r'pic.twitter.com/[\w]*',"", data)
    data = re.sub(r"\d+", "", data)
    data = data.replace(u'rt', '')
    data = data.translate(str.maketrans("","",string.punctuation))
    data = re.sub(r"\b[a-zA-Z]\b", "", data)
    return data.replace("http://", " ").replace("https://", " ")

def tokenizing (data):
    return word_tokenize(data)

def normalization(data):
    normalized_word = pd.read_csv("static/file/kamus/new_kamusalay.csv")
    normalizad_word_dict = {}
    for index, row in normalized_word.iterrows():
        if row[0] not in normalizad_word_dict:
            normalizad_word_dict[row[0]] = row[1]
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in data] 

def removed_stop_word(data):
    listStopword = stopwords.words('indonesian')
    listStopword.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                        'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                        'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                        'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                        'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                        'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                        '&amp', 'yah', 'gue', 'denny', 'jrx', 'pie', 
                        'jerinx', 'beuh', 'kp', 'ih', 'emang', 'bro', 'gitu',
                        'ane', 'donk', 'kok', 'sok', 'halah'])
    removed = []
    for t in data:
        if t not in listStopword:
            removed.append(t)
    return removed

def removed_stop_word_doc(words):
    list_stopwords = stopwords.words('indonesian')
    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah', 'gue', 'denny', 'jrx', 'pie', 
                       'jerinx', 'beuh', 'kp', 'ih', 'emang', 'bro', 'gitu',
                       'ane', 'donk', 'kok'])
    txt_stopword = pd.read_csv("static/file/kamus/stopword_list_tala.txt", names= ["stopwords"], header = None)
    list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
    list_stopwords = set(list_stopwords)
    return [word for word in words if word not in list_stopwords]

def stemming(data):
    string2=""
    for i in data:
        string2=string2+i+" "
    result = string2
    result = stemmer.stem(result)
    return result

def precision_neg(tabel_conf):
        precision_Neg = (tabel_conf[0][0] / (tabel_conf[0][0] + tabel_conf[1][0] + tabel_conf[2][0]))* 100.0
        return precision_Neg

def precision_neutral(tabel_conf):
        precision_Neu = (tabel_conf[1][1] / (tabel_conf[1][1] + tabel_conf[0][1] + tabel_conf[2][1]))* 100.0
        return precision_Neu

def precision_pos(tabel_conf):
        precision_Pos = (tabel_conf[2][2]/ (tabel_conf[2][2] + tabel_conf[1][2] + tabel_conf[0][2]))* 100.0
        return precision_Pos

def recall_neg(tabel_conf):
        recall_Neg = (tabel_conf[0][0] / (tabel_conf[0][0] + tabel_conf[0][1] + tabel_conf[0][2]))* 100.0
        return recall_Neg

def recall_neutral(tabel_conf):
        recall_Neu = tabel_conf[1][1] / (tabel_conf[1][1] + tabel_conf[1][0] + tabel_conf[1][2]) * 100.0
        return recall_Neu

def recall_pos(tabel_conf):
        recall_Pos = (tabel_conf[2][2]/ (tabel_conf[2][2] + tabel_conf[2][1] + tabel_conf[2][0]))* 100.0
        return recall_Pos

def f1_score(precision, recall):
    f1score = 2*((precision*recall)/(precision+recall))
    return f1score

def format(data):
    return "{:,.2f}".format(data)

@app.route('/')
def lexicon():
    processing = lexicon = []
    processing = open_file(processing_filepath)
    lexicon = open_file(lexicon_filepath)
    return render_template('lexicon.html', processing = processing, lexicon = lexicon, length = len(lexicon), pos = 8848, neg = 4196, neutral = 3844)

@app.route('/nbc')
def nbc():
    TWEET_DATA = pd.read_csv(lexicon_filepath, usecols=["preprocessing", "label"])
    TWEET_DATA.columns = ["preprocessing", "label"]

    def join_text_list(texts):
        texts = ast.literal_eval(texts)
        return ' '.join([text for text in texts])
    TWEET_DATA["tweet_join"] = TWEET_DATA["preprocessing"].apply(join_text_list)

    tf_idf = TfidfVectorizer(max_features=1000, binary=True)
    tfidf_mat = tf_idf.fit_transform(TWEET_DATA["tweet_join"]).toarray()

    x=tfidf_mat
    y=TWEET_DATA["label"]

    NBC = MultinomialNB()
    x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.2, random_state=2)

    #WITHOUT SMOTE
    without_smote  = NBC.fit(x_train, y_train)

    prediction_train = without_smote.predict(x_train)
    prediction_test = without_smote.predict(x_test)

    matrix_without_smote_train = confusion_matrix(y_train,prediction_train)
    matrix_without_smote_test = confusion_matrix(y_test,prediction_test)

    accuracy_training = accuracy_score(y_train, prediction_train)* 100.0
    accuracy_testing = accuracy_score(y_test,prediction_test)* 100.0
    precision_train_neg = precision_neg(matrix_without_smote_train)
    precision_train_neutral = precision_neutral(matrix_without_smote_train)
    precision_train_pos = precision_pos(matrix_without_smote_train)
    precision_test_neg = precision_neg(matrix_without_smote_test)
    precision_test_neutral = precision_neutral(matrix_without_smote_test)
    precision_test_pos = precision_pos(matrix_without_smote_test)
    recall_train_neg = recall_neg(matrix_without_smote_train)
    recall_train_neutral = recall_neutral(matrix_without_smote_train)
    recall_train_pos = recall_pos(matrix_without_smote_train)
    recall_test_neg = recall_neg(matrix_without_smote_test)
    recall_test_neutral = recall_neutral(matrix_without_smote_test)
    recall_test_pos = recall_pos(matrix_without_smote_test)
    f1score_train_neg = f1_score(precision_train_neg, recall_train_neg)
    f1score_train_neutral = f1_score(precision_train_neutral, recall_train_neutral)
    f1score_train_pos = f1_score(precision_train_pos, recall_train_pos)
    f1score_test_neg = f1_score(precision_test_neg, recall_test_pos)
    f1score_test_neutral = f1_score(precision_test_neutral, recall_test_neutral)
    f1score_test_pos = f1_score(precision_test_pos, recall_test_pos)

    #WITH SMOTE
    smote = SMOTE()
    x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)
    with_smote = NBC.fit(x_smote_train, y_smote_train)

    prediction_train_smote = with_smote.predict(x_smote_train)
    prediction_test_smote = with_smote.predict (x_test)

    matrix_smote_train = confusion_matrix(y_smote_train,prediction_train_smote)
    matrix_smote_test = confusion_matrix(y_test,prediction_test_smote)

    accuracy_training_smote = accuracy_score(y_smote_train, prediction_train_smote)* 100.0
    accuracy_testing_smote = accuracy_score(y_test,prediction_test_smote)* 100.0
    precision_train_smote_neg = precision_neg(matrix_smote_train)
    precision_train_smote_neutral = precision_neutral(matrix_smote_train)
    precision_train_smote_pos = precision_pos(matrix_smote_train)
    precision_test_smote_neg = precision_neg(matrix_smote_test)
    precision_test_smote_neutral = precision_neutral(matrix_smote_test)
    precision_test_smote_pos = precision_pos(matrix_smote_test)
    recall_train_smote_neg = recall_neg(matrix_smote_train)
    recall_train_smote_neutral = recall_neutral(matrix_smote_train)
    recall_train_smote_pos = recall_pos(matrix_smote_train)
    recall_test_smote_neg = recall_neg(matrix_smote_test)
    recall_test_smote_neutral = recall_neutral(matrix_smote_test)
    recall_test_smote_pos = recall_pos(matrix_smote_test)
    f1score_train_smote_neg = f1_score(precision_train_smote_neg, recall_train_smote_neg)
    f1score_train_smote_neutral = f1_score(precision_train_smote_neutral, recall_train_smote_neutral)
    f1score_train_smote_pos = f1_score(precision_train_smote_pos, recall_train_smote_pos)
    f1score_test_smote_neg = f1_score(precision_test_smote_neg, recall_test_smote_pos)
    f1score_test_smote_neutral = f1_score(precision_test_smote_neutral, recall_test_smote_neutral)
    f1score_test_smote_pos = f1_score(precision_test_smote_pos, recall_test_smote_pos)

    return render_template('nbc.html', all_data = len(y_train)+len(y_test), 
                            sentiment_training = y_train.value_counts(),
                            all_training = len(y_train), sentiment_testing = y_test.value_counts(),
                            all_testing = len(y_test), accuracy_training = format(accuracy_training), accuracy_testing = format(accuracy_testing),
                            precision_train_neg = format(precision_train_neg), precision_train_neutral = format(precision_train_neutral),
                            precision_train_pos = format(precision_train_pos), precision_test_neg = format(precision_test_neg),
                            precision_test_neutral = format(precision_test_neutral), precision_test_pos = format(precision_test_pos),
                            recall_train_neg = format(recall_train_neg), recall_train_neutral = format(recall_train_neutral),
                            recall_train_pos = format(recall_train_pos), recall_test_neg = format(recall_test_neg),
                            recall_test_neutral = format(recall_test_neutral), recall_test_pos = format(recall_test_pos), f1score_train_neg = format(f1score_train_neg),
                            f1score_train_neutral = format(f1score_train_neutral), f1score_train_pos = format(f1score_train_pos),
                            f1score_test_neg = format(f1score_test_neg), f1score_test_pos = format(f1score_test_pos), f1score_test_neutral = format(f1score_test_neutral),
                            matrix_without_smote_train = matrix_without_smote_train, matrix_without_smote_test = matrix_without_smote_test,
                            all_data_smote = len(y_smote_train)+len(y_test), sentiment_training_smote = y_smote_train.value_counts(),
                            all_training_smote = len(y_smote_train), sentiment_testing_smote = len(y_test), all_testing_smote = len(y_test),
                            accuracy_training_smote = format(accuracy_training_smote), accuracy_testing_smote = format(accuracy_testing_smote),
                            precision_train_smote_neg = format(precision_train_smote_neg), precision_train_smote_neutral = format(precision_train_smote_neutral),
                            precision_train_smote_pos = format(precision_train_smote_pos), precision_test_smote_neg = format(precision_test_smote_neg), 
                            precision_test_smote_neutral = format(precision_test_smote_neutral), precision_test_smote_pos = format(precision_test_smote_pos),
                            recall_train_smote_neg = format(recall_train_smote_neg), recall_train_smote_neutral = format(recall_train_smote_neutral),
                            recall_train_smote_pos = format(recall_train_smote_pos), recall_test_smote_neg = format(recall_test_smote_neg), recall_test_smote_neutral = format(recall_test_smote_neutral),
                            recall_test_smote_pos = format(recall_test_smote_pos), f1score_train_smote_neg = format(f1score_train_smote_neg),
                            f1score_train_smote_neutral = format(f1score_train_smote_neutral), f1score_train_smote_pos = format(f1score_train_smote_pos),
                            f1score_test_smote_neg = format(f1score_test_smote_neg), f1score_test_smote_neutral = format(f1score_test_smote_neutral),
                            f1score_test_smote_pos = format(f1score_test_smote_pos), matrix_smote_train = matrix_smote_train, matrix_smote_test = matrix_smote_test)

@app.route('/text')
def text():
    result = ""
    pred = ""
    return render_template('text.html', result = result, pred = pred)

@app.route('/download')
def download():
    return render_template('download.html')

@app.route('/document')
def document():
    table= []
    length = 0
    length_att = 0
    nama_file = ""
    return render_template('document.html', table = table, length = length, length_att = length_att, nama_file = nama_file)

@app.route('/text/result_text', methods= ['POST', 'GET'])
def predict():
    if request.method == 'POST':
        result = request.form['data']
        result = case_folding(result)
        result = cleansing(result)
        result = tokenizing(result)
        result = normalization(result)
        result = removed_stop_word(result)
        result = stemming(result)    
        pred = loaded_model.predict(loaded_vec.transform([result]))
        features = loaded_vec.get_feature_names()
        return render_template('text.html', pred = pred, result = result)
    else:
        return render_template('text.html')

@app.route('/show_document', methods= ['POST', 'GET'])
def show_doc():
    data = []
    if request.method == 'POST':
        if request.files:
            uploaded_file = request.files['doc_data']
            filepath = os.path.join(app.config['FILE_UPLOADS'], uploaded_file.filename)
            uploaded_file.save(filepath)
            nama_file = os.path.basename(filepath)
            nama_file = os.path.splitext(nama_file)[0]
            data = open_file(filepath)
    return render_template('document.html', table = data, length = len(data), length_att = len(data[0]),
                            filepath = filepath, nama_file = nama_file)

@app.route('/result_document', methods= ['POST', 'GET'])
def predict_doc():
    if request.method == 'POST':
        attribute = request.form['attribute']
        label = request.form['label']
        hidden = request.form['hidden']
        nama_file = request.form['nama_file']

        data = pd.read_csv(hidden,  error_bad_lines=False)
        attribute = attribute.replace(u'\ufeff', '')
        label = label.replace(u'\ufeff', '')

        df = pd.DataFrame(data[[attribute, label]])

        df["case_folding"] = df[attribute].apply(case_folding)

        def remove_pattern(data, pattern):
            r = re.findall(pattern, data)
            for i in r:
                data = re.sub(i, '', data)
            return data
        df["remove_user"] = np.vectorize(remove_pattern)(df["case_folding"], "@[\w]*")
        df["cleansing"] = df["remove_user"].apply(cleansing)
        df["tokenizing"] = df["cleansing"].apply(tokenizing)
        df["normalization"] = df["tokenizing"].apply(normalization)
        df['remove_stop_word'] = df["normalization"].apply(removed_stop_word_doc) 

        def stemmed_wrapper(term):
            return stemmer.stem(term)
        term_dict = {}
        for document in df['remove_stop_word']:
            for term in document:
                if term not in term_dict:
                    term_dict[term] = ' '
        for term in term_dict:
            term_dict[term] = stemmed_wrapper(term)
        def stemming(document):
            return [term_dict[term] for term in document]
        df['stemming'] = df['remove_stop_word'].apply(stemming)
        df.to_csv("static/file/processing/%s_processing.csv" %nama_file)
        
        x_test = df['stemming'].apply(', '.join)
        x_test = x_test.replace(','," ")
        for data_word_cloud in x_test:
            data_word_cloud = str(data_word_cloud)
        x_test = loaded_vec.transform(x_test)
        y_test = df[label]
        prediction = loaded_model.predict(x_test)

        tabel_conf = confusion_matrix(y_test,prediction)

        accuracy = accuracy_score(y_test,prediction)* 100.0
        accuracy = format(accuracy)

        precision_negatif = precision_neg(tabel_conf)
        precision_netral = precision_neutral(tabel_conf)
        precision_positif = precision_pos(tabel_conf)
        recall_negatif = recall_neg(tabel_conf)
        recall_netral = recall_neutral(tabel_conf)
        recall_positif = recall_pos(tabel_conf)
        f1_score_neg = f1_score(precision_negatif, recall_negatif)
        f1_score_netral = f1_score(precision_netral, recall_netral)
        f1_score_pos = f1_score(precision_positif, recall_positif)

        word_cloud = wc.WordCloud(collocations=False, background_color='white').generate(data_word_cloud)
        plt.figure(figsize=[20,10], facecolor='#f8f9fd')
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig('static/file/wordcloud/%s_wordcloud.jpg' %nama_file)

    return render_template('result-document.html', column_names=df.columns.values, row_data = list(df.values.tolist()),
                            zip = zip, accuracy = accuracy, precision_negatif = format(precision_negatif), precision_netral = format(precision_netral),
                            precision_positif = format(precision_positif), recall_negatif =  format(recall_negatif), recall_netral = format(recall_netral),
                            recall_positif = format(recall_positif), f1_score_neg = format(f1_score_neg), f1_score_netral = format(f1_score_netral),
                            f1_score_pos = format(f1_score_pos),tabel_conf = tabel_conf, nama_file = nama_file, prediction = prediction)

app.config['FILE_UPLOADS'] = "static\\file\\uploads"

if __name__ == '__main__':
    app.run(debug=True)

