# save_vectors.py - saves cache of computed vectors for sentences

import gensim
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re

input_file = '../data/input/hist_tweets.csv'
google_file = '/Users/zubo/Datasets/Word2Vec/GoogleNews-vectors-negative300.bin'
output_file = '../data/output/tweets-vec.csv'

df = pd.read_csv(input_file)
print('Total {} tweets'.format(len(df)))

sentences = [t.text.lower().split() for index, t in df.iterrows()]


def text_to_words(t):
    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", t)

    # Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # Modify stop words
    stops = set(stopwords.words("english"))
    add_stops = set(['rt', 'http', 'https'])
    stops = stops.union(add_stops)

    # Remove stop words
    meaningful_words = [w for w in words if w not in stops]

    return(meaningful_words)


df['sentences'] = df.apply(lambda x: text_to_words(x.text), axis=1)
sentences = df.sentences.tolist()
model = gensim.models.KeyedVectors.load_word2vec_format(google_file,
                                                        binary=True)

vectors = list()
for sentence in sentences:
    vectors.append([model[word] for word in sentence
                    if word in model])

avg_vectors = [np.mean(sentence_vectors, axis=0)
               for sentence_vectors in vectors]
df['avg_vector'] = avg_vectors


# Save vectors for later
df.to_csv(output_file, index=False)

print('Vectors saved in a file {}'.format(output_file))
