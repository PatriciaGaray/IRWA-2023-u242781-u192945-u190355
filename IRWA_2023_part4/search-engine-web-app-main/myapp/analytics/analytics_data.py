import json
import random
from collections import Counter


class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    # statistics table 1
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    fact_clicks = dict([])

    # statistics table 2
    fact_two = dict([])

    # statistics table 3
    fact_three = dict([])

    query_terms = []

    # def save_query_terms(self, terms: str) -> int:
    #     print(self)
    #     return random.randint(0, 100000)

    def save_query_terms(self, terms: str) -> int:
        self.query_terms.append(terms)
        return len(self.query_terms) 
    
    def get_query_frequency(self):
        # Use Counter to count the frequency of each query term
        query_counter = Counter(self.query_terms)

        # Separate the queries and their frequencies
        query_labels = list(query_counter.keys())
        query_data = list(query_counter.values())

        return query_labels, query_data

class ClickedDoc:
    def __init__(self, id, tweet, counter):
        self.doc_id = id
        self.tweet = tweet
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
