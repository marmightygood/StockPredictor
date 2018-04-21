import tweepy
import sys

consumer_key = sys.argv[1]
consumer_secret = sys.argv[2]
access_token = sys.argv[3]
access_token_secret = sys.argv[4]

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

org_name = "Fletcher Building Ltd"
org_name_tokens = org_name.split(' ')
query = ' OR '.join (org_name_tokens) 


max_tweets = 10
searched_tweets = [status for status in tweepy.Cursor(api.search, q=query).items(max_tweets)]

print (searched_tweets)