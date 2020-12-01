import csv  # read excel
import random

import nltk
from nltk.corpus import stopwords
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from scipy import spatial
from autocorrect import Speller  # Check spelling
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from google_trans_new import google_translator  # translate
import spacy  # Analyze the word components in the sentence, the efficiency is higher than nltk.pos_tag

nlp = spacy.load("en_core_web_sm")
translator = google_translator()


def translate_query(sentence):
    """
    Google has changed the way the token is created. Sometimes the api may cause AttributeError
    :param sentence: user query
    :return: translate sentence
    """
    try:
        query = translator.translate(sentence, lang_tgt='en')
        return query
    except Exception:
        return False


def query_optimizer(sentence):
    """
    The first processing of incoming queries eliminates spelling errors, null input, and other language problems.
    :param sentence: user input
    :return: string which optimize or Null
    """
    if sentence.strip() == "":
        return "Null"
    optimizer = False
    while not optimizer:  # There is a hidden danger here, Google Translate sometimes becomes unresponsive
        optimizer = translate_query(sentence)
    check = Speller(lang='en')
    return check(optimizer)


def normalization(text, stop_words=True):
    """
    Standardized sentences
    :param text: input
    :param stop_words: English stop word
    :return: sentence with stemmed , lower and remove number and stopwords
    """
    tokenizer = nltk.RegexpTokenizer(r"[a-zA-Z]\w+")
    tok_document = tokenizer.tokenize(text)
    documents = []
    s = " "
    if stop_words is True:
        english_stopwords = stopwords.words('english')
    else:
        english_stopwords = ['what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how']
    for word in tok_document:
        if word.lower() not in english_stopwords:
            documents.append(word.lower())
    if not documents:
        return "Excessive deletion"
    sb_stemmer = SnowballStemmer('english')
    stemmed_documents = [sb_stemmer.stem(word) for word in documents]
    s = s.join(stemmed_documents)
    return s


def build_dataset(file):
    """
    build dic
    :return: The question is the key, the answer is the dictionary of the value
    """
    data = []
    labels = []
    book = []
    question_map = {}
    with open(file, encoding='utf8', errors='ignore', mode="r") as csv_file:
        csv_reader = csv.reader(csv_file)  # Use csv.reader to read the file in csv_file
        for row in csv_reader:
            labels.append(row[1])
            data.append(row[2])
    for index in range(1, len(labels)):
        if labels[index] != labels[index - 1]:
            book.clear()
        book.append(data[index])
        b = book.copy()  # Shallow copy
        question_map[labels[index]] = b
    return question_map


def get_tag(file, tag, data_book=None, label_book=None):
    """
    help classifier
    :param file:
    :param tag:
    :param data_book:
    :param label_book:
    :return:
    """
    if label_book is None:
        label_book = []
    if data_book is None:
        data_book = []
    need_part = build_dataset(file).keys()
    for part in need_part:
        data_book.append(part)
        label_book.append(tag)
    return data_book, label_book


def check_classifier():
    """
    make a classifier to recognize the type of input
    :return: LogisticRegression classifier and two object
    """
    content = []
    labels = []
    file = 'COMP3074-CW1-Dataset.csv'
    content, labels = get_tag(file, "question_book", content, labels)
    file = 'name.csv'
    content, labels = get_tag(file, "question_book", content, labels)
    file = 'Small_talk.csv'
    content, labels = get_tag(file, "small_talk", content, labels, )
    x_train, x_test, y_train, y_test = train_test_split(content,  # Sample feature set to be divided
                                                        labels,  # The sample result to be divided (label)
                                                        stratify=labels,  # Keep the category proportions
                                                        # the same in training and testing
                                                        test_size=0.25,  # Refers to the proportion of
                                                        # samples reserved for testing
                                                        random_state=22)  # Random seed
    count_vect = CountVectorizer(stop_words=stopwords.words('english'))
    x_train_counts = count_vect.fit_transform(x_train)
    tfidf_transformer = TfidfTransformer(use_idf=True,  # Tf_idf
                                         sublinear_tf=True).fit(x_train_counts)
    x_train_tf = tfidf_transformer.transform(x_train_counts)  # Standardize the inherent attributes of the training set,
    # reduce dimensionality and normalize
    classify = LogisticRegression(random_state=0).fit(x_train_tf, y_train)  # Logistic regression
    return classify, tfidf_transformer, count_vect


def modeling(matrix):
    """
    model countvectorizer and fit
    :param matrix: matrix which use to fit
    :return: countvectorizer and countvectorizer_fit
    """
    cv = CountVectorizer()
    cv_fit = cv.fit_transform(matrix)
    return cv, cv_fit


def set_key(dic, container, check=True):
    """
    Make a question training set
    :param dic:data dic
    :param container: a question set helps to use different document
    :param check: whether the need to use stop words
    :return: question set
    """
    all_text = dic.keys()  # Only training questions
    for text in all_text:
        key = normalization(text, check)
        if key == "Excessive deletion":
            key = normalization(text, False)
        container.append(key)
    return container


def solve(cv, cv_fit, ask):
    """
    use tfidf model to find answer
    :param cv:countvertize
    :param cv_fit:countvertize_fit
    :param ask:query which modified
    :return: the index of the dic and accuracy of the cos
    """
    cv_trans = cv.transform(ask)
    tfidf = TfidfTransformer(use_idf=True,
                             sublinear_tf=True).fit(cv_fit)
    tfidf_fit = tfidf.transform(cv_fit).toarray()
    tfidf_trans = tfidf.transform(cv_trans).toarray()
    tfbook = []
    for i in range(len(tfidf_fit)):
        sim_1 = 1 - spatial.distance.cosine(tfidf_trans[0], tfidf_fit[i])
        tfbook.append(sim_1)
    np.array(tfbook)
    index = np.argmax(tfbook)
    return index, tfbook[index]


def response(cv, cv_fit, dic, query):
    """
    the response of small_talk
    :param cv: countvectorizer
    :param cv_fit: countvectorizer_fit
    :param dic: data dictionary
    :param query: query which normalization
    """
    index, accuracy = solve(cv, cv_fit, query)
    book = list(dic)
    answer = dic[book[index]]
    if check_accuracy(accuracy):
        print("Sorry, I don’t understand what you mean, could you please describe it more clearly")
    elif accuracy < 0.6:
        print(f"Do you ask {book[index]}? I only know {answer[0]}")
    elif len(answer) > 1:
        print(answer[random.randint(0, len(answer) - 1)])
    else:
        print(answer[0])


def check_accuracy(number):
    """
    check accuracy
    :param number: accuracy
    :return: if no match, return false
    """
    if number != number:
        return True
    return False


if __name__ == "__main__":
    name = None
    rep = []
    response_book = []
    key_book = []
    small_talk_book = []
    dataset_file = 'COMP3074-CW1-Dataset.csv'
    small_talk_file = 'Small_talk.csv'
    name_file = 'name.csv'
    dataset = build_dataset(dataset_file)
    name_dic = build_dataset(name_file)
    small_talk_dic = build_dataset(small_talk_file)
    key_book = set_key(dataset, key_book)
    key_book = set_key(name_dic, key_book, False)
    dataset.update(name_dic)
    small_talk_book = set_key(small_talk_dic, small_talk_book)
    question_cv, question_cv_fit = modeling(key_book)
    small_talk_cv, small_talk_cv_fit = modeling(small_talk_book)
    classifier, tf_idf, count = check_classifier()
    stop = False
    while not stop:
        query = input("Enter your query, or STOP to quit, and press return:")
        if query == "STOP" or query == "quit":
            stop = True
        elif query == "":
            print("Sorry, I don’t understand what you mean, could you please describe it more clearly")
        else:
            query = query_optimizer(query)
            query_nor = [query]
            tip = classifier.predict(tf_idf.transform(count.transform(query_nor)))
            print(f'You are searching for {query}')
            question = normalization(query)
            if question.count(' ') <= 1:
                question = normalization(query, False)
            if question == "null":
                print("Sorry, I don’t understand what you mean, could you please describe it more clearly")
            else:
                query_nor = [question]
                if tip[0] == "question_book":
                    index, accuracy = solve(question_cv, question_cv_fit, query_nor)
                    book = list(dataset)
                    answer = dataset[book[index]]
                    if check_accuracy(accuracy):
                        print("Sorry, I don’t understand what you mean, could you please describe it more clearly")
                    elif accuracy < 0.6:
                        print(f"Do you ask {book[index]}? I only know {answer[0]}")
                    elif len(answer) > 1:
                        print(answer[random.randint(0, len(answer) - 1)])
                    else:
                        if answer[0] == "1":
                            if name:
                                print(f"Hi,{name[0]}. Nice to meet you!")
                            else:
                                print("Sorry sir, I don't know your name. Would you like to tell me?")
                        elif answer[0] == "2":
                            doc = nlp(query)
                            for token in doc:
                                if token.tag_ == 'NNP':
                                    rep.append(token)
                            if rep:
                                if len(rep) == 1:
                                    name = rep.copy()
                                    rep.clear()
                                    print(f"Hi,{name[0]}. Nice to meet you!")
                                else:
                                    rep.clear()
                                    print("Sorry sir, I get too many names. Could you type your name again？")
                            else:
                                print("Sorry sir, I can't get your name. Would you please type your name officially?")
                        else:
                            print(answer[0])
                else:
                    response(small_talk_cv, small_talk_cv_fit, small_talk_dic, query_nor)

"""
Q1: Hello, how are you?
Q2: What's my name?
Q3: How many people play in a Hockey team?
Q4: How many people live in Atlantis?
Q5: What are the big ten?
Q6: What is single malt scotch?
Q7: Who is Isaac Newton?
Q8: What is the best Dim Sum?
Q9: What is mustard made of?
Q10: What is 奶茶？
"""
