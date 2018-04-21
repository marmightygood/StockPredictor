import tweepy
consumer_key="MRCh0g6rsu97B8Q1mYyJcQdPM"
consumer_secret="66gNlUa3KyFy0LYz1W6w0qfdmNydzHuj6G5ePFAenWAiyNq69o"
access_token="2725621934-kxI08KpuL5350ZXoKUaBBnwhSCJEtcWMZzTwa8r"
access_token_secret="Gcyu19jL6M0Lc4zka2psfuDoczqgTlW59P4ilbQeTNCzS"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

org_name = "Fletcher Building Ltd"
org_name_tokens = org_name.split(' ')
query = ' OR '.join (org_name_tokens) 


max_tweets = 10
searched_tweets = [status for status in tweepy.Cursor(api.search, q=query).items(max_tweets)]

print (searched_tweets)