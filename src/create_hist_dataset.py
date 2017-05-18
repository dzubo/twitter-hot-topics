import pandas as pd
import pickle

acc_list_file = '../data/input/acc_list.csv'
tweets_pickle_file = '../data/input/tweets.pickle'
output = '../data/output/hist_tweets.csv'


print('Load tweets from {}'.format(tweets_pickle_file))
with open(tweets_pickle_file, 'rb') as f:
    tweets = pickle.load(f)
print('{} tweets loaded'.format(len(tweets)))

print('Extracting text and attributes.')
tweets_dic = [tw.AsDict() for tw in tweets]
df = pd.DataFrame(tweets_dic)

ids = df.apply(lambda x: x.user['id'], axis=1)
cols = ['created_at', 'favorite_count', 'id', 'lang',
        'retweet_count', 'text', 'user']
df['user'] = ids
df = df[cols]

df.to_csv(output, index=False)
print('Dataset saved into {}'.format(output))
