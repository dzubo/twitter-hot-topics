# get_tweets.py

import os
import twitter
import pandas as pd
import datetime
import pickle

if ('CONSUMER_KEY' in os.environ):
    CONSUMER_KEY = os.environ['CONSUMER_KEY']
    print('CONSUMER_KEY set')
else:
    print("Environment variable CONSUMER_KEY is not specified")

if ('CONSUMER_SECRET' in os.environ):
    CONSUMER_SECRET = os.environ['CONSUMER_SECRET']
    print('CONSUMER_SECRET set')
else:
    print("Environment variable CONSUMER_SECRET is not specified")

if ('ACCESS_TOKEN' in os.environ):
    ACCESS_TOKEN = os.environ['ACCESS_TOKEN']
    print('ACCESS_TOKEN set')
else:
    print("Environment variable ACCESS_TOKEN is not specified")

if ('ACCESS_TOKEN_SECRET' in os.environ):
    ACCESS_TOKEN_SECRET = os.environ['ACCESS_TOKEN_SECRET']
    print('ACCESS_TOKEN_SECRET set')
else:
    print("Environment variable ACCESS_TOKEN_SECRET is not specified")


acc_list_file = '../data/input/acc_list.csv'
INIT_DEPTH_DAYS = 60
CHUNK_SIZE = 200

# Create an Api instance.
api = twitter.Api(consumer_key=CONSUMER_KEY,
                  consumer_secret=CONSUMER_SECRET,
                  access_token_key=ACCESS_TOKEN,
                  access_token_secret=ACCESS_TOKEN_SECRET)

# users = api.GetFriends()
acc_list = pd.read_csv(acc_list_file, index_col=False)


# returns list of last tweets, or last tweets with id less than max_id
def get_chunk_tweets(user_id, chunk_size=200, max_id=None):
    tweets = api.GetUserTimeline(user_id=user_id,
                                 count=chunk_size, trim_user=True,
                                 exclude_replies=True,
                                 max_id=max_id)
    return(tweets)


def get_created_at(tweet):
    dt = pd.to_datetime(tweet.created_at) \
        .to_pydatetime()
    return(dt)


def get_tweets_from_user(user_id):
    now = datetime.datetime.now()
    from_dt = now - datetime.timedelta(days=INIT_DEPTH_DAYS)

    tweets = get_chunk_tweets(user_id, CHUNK_SIZE)
    if(len(tweets) == 0):
        print('No tweets loaded')
        return(list())
    min_id = tweets[-1].id
    newest_dt = get_created_at(tweets[0])
    oldest_dt = get_created_at(tweets[-1])
    print('Downloaded {} tweets from {} to {}'
          .format(len(tweets), oldest_dt, newest_dt))
    counter = 50

    while((oldest_dt > from_dt) & (counter > 0)):
        counter += -1
        more_tweets = get_chunk_tweets(user_id, CHUNK_SIZE,
                                       max_id=min_id - 1)
        if(len(more_tweets) == 0):
            break

        tweets += more_tweets
        min_id = more_tweets[-1].id
        newest_dt = get_created_at(more_tweets[0])
        oldest_dt = get_created_at(more_tweets[-1])
        print('Downloaded {} tweets from {} to {}'
              .format(len(more_tweets), oldest_dt, newest_dt))
    return(tweets)


all_tweets = list()
for index, row in acc_list.iterrows():
    print('Downloading tweets from {} ({}/{})'
          .format(row.screen_name, index + 1, len(acc_list)))
    tweets = get_tweets_from_user(row.user_id)
    all_tweets += tweets

# Save in binary format
with open('tweets.pickle', 'wb') as f:
    pickle.dump(all_tweets, f, protocol=-1)

# Save as json file
all_tweets_json = [tweet.AsJsonString() for tweet in all_tweets]
with open('tweets.json', 'w') as f:
    for tweet in all_tweets_json:
        f.write(tweet + '\n')
