# Libraries for text preprocessing
import os

import nltk

#nltk.download('stopwords')
from nltk.corpus import stopwords

#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

#nltk.download('averaged_perceptron_tagger')
#nltk.download('tagsets')
# Word cloud

import re

# Barplot of most freq words

if not os.path.exists('out/'):
    os.makedirs('out/')

stop_words = set(stopwords.words("english"))


def extractKeyWords(text):
    tagged_sentence = nltk.tag.pos_tag(text.split())
    edited_sentence = [word for word, tag in tagged_sentence if
                       tag not in ['NNP', 'NNPS', 'CC', 'CD', 'DT', 'LS', 'POS']]
    text = ' '.join(edited_sentence)

    # Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    # remove stop words
    words = text.split()
    words = [word for word in words if not word in stop_words]
    text = " ".join(words)

    # Lemmatisation
    lem = WordNetLemmatizer()
    words = text.split()
    words = [lem.lemmatize(word) for word in words]
    text = " ".join(words)

    # Stemming
    ps = PorterStemmer()
    words = text.split()
    words = [ps.stem(word) for word in words]
    text = " ".join(words)

    return text

