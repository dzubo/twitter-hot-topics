# Twitter Hot Topics Detection

A small project to demonstrate the usage of Twitter API and NLP techniques.
The idea is to download tweets from specified accounts (news companies),
cluster tweets into topics, detect the hottest topic, and output the most relevant news tweet from that topic.  

This is not production-ready code, more like a proof of concept.

## Concepts
The project was done using the following tools and techiques:

* [Twitter API](https://dev.twitter.com/overview/api) ([python-twitter](https://github.com/bear/python-twitter) implementation)
* [Google Word2Vec](https://code.google.com/archive/p/word2vec/) feature generation ([pre-trained vectors](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) trained on part of Google News dataset)
* [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) ([sklearn.cluster.KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html))
* [Silhouette values](https://en.wikipedia.org/wiki/Silhouette_(clustering)) to estimate the number of clusters ([sklearn.metrics.silhouette_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html))

## Files

The code split into separate files to make debugging and testing easier.

[get_tweets.py](src/get_tweets.py) - downloads news tweets.  
[create_hist_dataset.py](src/create_hist_dataset.py) - cleans and saves dataset.  
[save_vectors.py](src/save_vectors.py) - converts sentences to vectors and save result for further modelling.  
[detect_hot.py](src/detect_hot.py) - prints out 'hot' tweets.

## Some (dirty) exploration

There is some thought process recorded in the Jupyter Notebooks.

[NLP explore.ipynb](notebooks/NLP%20explore.ipynb) - some exploration on clustering.  
[Tune parameters.ipynb](src/Tune%20parameters.ipynb) - some exploration on tuning heuristic parameters.  