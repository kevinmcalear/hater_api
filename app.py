# -*- coding: utf-8 -*-
# Getting the guts of our app
from flask import Flask, render_template, request, jsonify
# Getting things to come back from json to a dict
import json
# Making that json pretty if needed
from pprint import pprint
# Dealing with stupid Unicode issues
from django.utils.encoding import smart_str
# Bring me my Pickles, slave!
from sklearn.externals import joblib
# Importing my standard stuff
import pandas as pd
import numpy as np
# For troubleshooting
# Use This: code.interact(local=locals())
import code


# Get all a users comments and run them through my model
def user_score(comments, my_vect, clf):
    comments = filter(None, comments)
    badwords = set(pd.read_csv('data/my_badwords.csv').words)
    badwords_count = []

    for el in comments:
        tokens = el.split(' ')
        badwords_count.append(len([i for i in tokens if i.lower() in badwords]))

    n_words = [len(c.split()) for c in comments]
    allcaps = [np.sum([w.isupper() for w in comment.split()]) for comment in comments]
    allcaps_ratio = np.array(allcaps) / np.array(n_words, dtype=np.float)
    bad_ratio = np.array(badwords_count) / np.array(n_words, dtype=np.float)
    exclamation = [c.count("!") for c in comments]
    addressing = [c.count("@") for c in comments]
    spaces = [c.count(" ") for c in comments]

    re_badwords = np.array(badwords_count).reshape((len(badwords_count),1))
    re_n_words = np.array(n_words).reshape((len(badwords_count),1))
    re_allcaps = np.array(allcaps).reshape((len(badwords_count),1))
    re_allcaps_ratio = np.array(allcaps_ratio).reshape((len(badwords_count),1))
    re_bad_ratio = np.array(bad_ratio).reshape((len(badwords_count),1))
    re_exclamation = np.array(exclamation).reshape((len(badwords_count),1))
    re_addressing = np.array(addressing).reshape((len(badwords_count),1))
    re_spaces = np.array(spaces).reshape((len(badwords_count),1))

    vect = my_vect.transform(comments)
    features = np.hstack((vect.todense(), re_badwords))
    features = np.hstack((features, re_n_words))
    features = np.hstack((features, re_allcaps))
    features = np.hstack((features, re_allcaps_ratio))
    features = np.hstack((features, re_bad_ratio))
    features = np.hstack((features, re_exclamation))
    features = np.hstack((features, re_addressing))
    features = np.hstack((features, re_spaces))
    predictions = clf.predict_proba(features)

    return predictions


# Setting up app
app = Flask(__name__)

# ************ DEBUG CRAP ************
# log to stderr
import logging
from logging import StreamHandler
file_handler = StreamHandler()
app.logger.setLevel(logging.DEBUG)  # set the desired logging level here
app.logger.addHandler(file_handler)
# ************ DEBUG CRAP ************

print 'Loading clf & vect...'
vect = joblib.load('vect.pkl')
clf = joblib.load('clf.pkl')
print 'All loaded Captn\'!'

# Setting up our base route
@app.route('/')
def display_form():
    return render_template('layout.html')

# Score an individual comment
@app.route('/score-comment', methods=['GET'])
def score_comment():
    comment = request.args.get('comment').encode('utf-8')
    print 'this was the comment', comment
    score = user_score([comment], vect, clf)
    return jsonify({'score': score[0][1]})

# score a group of comments
@app.route('/score-comments', methods=['GET'])
def score_comments():
    comments = request.args.get('comments').split(',')
    comments = list(map((lambda comment: smart_str(comment)), comments))
    print 'this was the json', comments
    score = user_score(comments, vect, clf)
    print 'SCORE:', score
    mapped_score = list(map((lambda x: x[1]), score))
    return jsonify({'scores': mapped_score})

if __name__ == '__main__':
    app.debug = True
    app.run()
