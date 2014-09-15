import httplib
import urllib


def http_post(url, api_key, content, extra_headers={}):
    conn = httplib.HTTPConnection("localhost", 8000)
    headers = {}
    headers.update(extra_headers)
    headers['x-api-key'] = api_key
    conn.request("POST", url, content, headers=headers)
    return conn.getresponse()


def dataset_to_str(dataset):
    s = ''
    for line in dataset:
        s += ','.join(str(i) for i in line)
        s += '\n'
    return s


class BaseModel(object):
    URL_TPL = None
    URL_CREATE_TPL = None

    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name

    def predict(self, pattern):
        p = ','.join(unicode(p) for p in pattern)

        url = self.URL_TPL % self.model_name
        response = http_post(url, self.api_key, p)
        return response.read()


class Classifier(BaseModel):
    URL_TPL = '/api/classifier/%s/predict'

    def create(self, dataset):
        content = dataset_to_str(dataset)
        data = dict(name=self.model_name, data=content)
        data = urllib.urlencode(data)

        url = '/api/classifier/'
        h = {}
        h['Content-type'] = 'application/x-www-form-urlencoded'
        h['Accept'] = 'text/plain'

        response = http_post(url, self.api_key, data, h)
        if response.status != 200:
            raise ValueError(response.read())


class Grouper(BaseModel):
    URL_TPL = '/api/grouper/%s/predict'

    def create(self, dataset, clusters):
        content = dataset_to_str(dataset)
        data = dict(name=self.model_name, data=content, clusters=clusters)
        data = urllib.urlencode(data)

        url = '/api/grouper/'
        h = {}
        h['Content-type'] = 'application/x-www-form-urlencoded'

        response = http_post(url, self.api_key, data, h)
        if response.status != 200:
            raise ValueError(response.read())
