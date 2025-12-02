from lxml import etree
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import fasttext.util
from sklearn.svm import LinearSVC

fasttext.util.download_model('en',if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')

def embed_fasttext(text):
    return ft.get_sentence_vector(text)

def build_fasttext_features(df):
    X = np.vstack([embed_fasttext(t) for t in df["input"]])
    y = df["polarity"].values
    return X, y

def train_fasttext_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LinearSVC()
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("\n===== FastText + SVM =====")
    print(classification_report(y_test, preds))

    return clf

def load_semeval_xml(path):
    tree = etree.parse(path)
    root = tree.getroot()

    data = []

    for sentence in root.findall("sentence"):
        sent_id = sentence.get("id")
        text = sentence.find("text").text.strip()

        aspects = []
        aspect_terms = sentence.find("aspectTerms")

        if aspect_terms is not None:
            for term in aspect_terms.findall("aspectTerm"):
                aspects.append({
                    "term": term.get("term"),
                    "polarity": term.get("polarity"),
                    "from": int(term.get("from")),
                    "to": int(term.get("to"))
                })

        data.append({
            "id": sent_id,
            "text": text,
            "aspects": aspects
        })

    return data

def build_apc_dataset(parsed_xml):
    rows = []

    for item in parsed_xml:
        sent = item["text"]
        for asp in item["aspects"]:
            rows.append({
                "sentence": sent,
                "aspect": asp["term"],
                "polarity": asp["polarity"]
            })

    return pd.DataFrame(rows)


def prepare_text(df):
    df["input"] = df["aspect"] + " [SEP] " + df["sentence"]
    return df

df = build_apc_dataset(load_semeval_xml("Laptop_Train_v2.xml"))  
df = prepare_text(df)

def train_tfidf_lr(df):
    X = df["input"]
    y = df["polarity"]

    tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=10000,
        sublinear_tf=True
    )

    X_vec = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("\n===== TF-IDF + Logistic Regression =====")
    print(classification_report(y_test, preds))

    return clf, tfidf

clf, tfidf = train_tfidf_lr(df)
X, Y = build_fasttext_features(df)
train_fasttext_svm(X,Y)