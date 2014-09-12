import httplib


def http_post(url, api_key, content):
    conn = httplib.HTTPConnection("localhost", 8000)
    headers = {}
    headers['x-api-key'] = api_key
    conn.request("POST", url, content, headers=headers)
    return conn.getresponse()


class Classifier(object):
    URL_TPL = '/api/classifier/%s/predict'

    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name

    def predict(self, pattern):
        p = ','.join(unicode(p) for p in pattern)

        url = self.URL_TPL % self.model_name
        response = http_post(url, self.api_key, p)
        return response.read()
