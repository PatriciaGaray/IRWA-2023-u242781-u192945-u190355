import json


class Document:
    """
    Original corpus data as an object
    """

    def __init__(self, id, title, tweet, date, hashtags, likes, retweets, url):
        self.id = id
        self.title = title
        self.tweet = tweet
        self.date = date
        self.hashtags = hashtags
        self.likes = likes
        self.retweets = retweets
        self.url = url

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)

class ResultItem:
    def __init__(self, id, title, tweet, date, hashtags, likes, retweets, url, ranking):
        self.id = id
        self.title = title
        self.tweet = tweet
        self.date = date
        self.hashtags = hashtags
        self.likes = likes
        self.retweets = retweets
        self.url = url
        self.ranking = ranking

class DocumentInfo:
    def __init__(self, id, title, tweet, date, hashtags, likes, retweets, url, ranking):
        self.id = id
        self.title = title
        self.tweet = tweet
        self.date = date
        self.hashtags = hashtags
        self.likes = likes
        self.retweets = retweets
        self.url = url
        self.ranking = ranking


class StatsDocument:
    """
    Original corpus data as an object
    """

    def __init__(self, id, title, tweet, date, count):
        self.id = id
        self.title = title
        self.tweet = tweet
        self.date = date
        self.count = count

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
