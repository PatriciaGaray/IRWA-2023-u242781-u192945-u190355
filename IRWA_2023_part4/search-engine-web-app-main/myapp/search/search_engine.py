import random

from myapp.search.objects import ResultItem, Document, DocumentInfo
from myapp.search.algorithms import search_in_corpus

def build_demo_results(corpus: dict, search_id):
    """
    Helper method, just to demo the app
    :return: a list of demo docs sorted by ranking
    """
    res = []
    size = len(corpus)
    ll = list(corpus.values())
    for index in range(random.randint(0, 40)):
        item: Document = ll[random.randint(0, size)]
        res.append(ResultItem(item.id, item.title, item.tweet, item.date, item.hashtags, item.likes, item.retweets, item.url, "doc_details?id={}&search_id={}&param2=2".format(item.id, search_id), random.randint(0, size)))

    #for index, item in enumerate(corpus['Id']):
    #     # DF columns: 'Id' 'Tweet' 'Username' 'Date' 'Hashtags' 'Likes' 'Retweets' 'Url' 'Language'
         #res.append(DocumentInfo(item.id, item.tweet, item.date, item.hashtags, item.likes, item.retweets, item.url,"doc_details?id={}&search_id={}&param2=2".format(item.id, search_id), random.random()))

    # simulate sort by ranking
    res.sort(key=lambda doc: doc.ranking, reverse=True)
    return res


class SearchEngine:
    """educational search engine"""

    def search(self, search_query, search_id, corpus: dict, search_option):
        print("Search query:", search_query)

        results = []
        ##### your code here #####
        search_results = search_in_corpus(search_query, corpus, search_option)
        
        # Transformar los resultados para que coincidan con ResultItem
        for item_id in search_results:
            doc = corpus[item_id]
            # Aquí debes adaptar los campos 'tweet', 'date', 'hashtags', 'likes', 'retweets' y 'url' según tu estructura de datos.
            # El valor random para 'ranking' se genera aquí, pero podría ser necesario ajustarlo dependiendo de la lógica de ranking real.
            result_item = ResultItem(doc.id, doc.title, doc.tweet, doc.date, doc.hashtags, doc.likes, doc.retweets, "doc_details?id={}&search_id={}&param2=2".format(doc.id, search_id), random.randint(0, len(corpus)))
            results.append(result_item)
        ##### your code here #####

        return results
