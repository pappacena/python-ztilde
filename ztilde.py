import httplib
import urllib
import json


HOST = 'ztilde.com'
PORT = 80


def http_post(url, api_key, content, extra_headers={}):
    conn = httplib.HTTPConnection(HOST, PORT)
    headers = {}
    headers.update(extra_headers)
    headers['x-api-key'] = api_key
    conn.request("POST", url, content, headers=headers)
    return conn.getresponse()


def http_get(url, api_key):
    conn = httplib.HTTPConnection(HOST, PORT)
    headers = {}
    headers['x-api-key'] = api_key
    conn.request("GET", url, '', headers)
    return conn.getresponse()


def dataset_to_str(dataset):
    s = ''
    for line in dataset:
        s += ','.join(str(i) for i in line)
        s += '\n'
    return s


def get_ztildes(api_key):
    resp = http_get('/api/model/', api_key)
    if resp.status != 200:
        raise ValueError('HTTP Request error: %s' % resp.read())
    data = json.loads(resp.read())

    retlist = []
    for c in data['clustering']:
        retlist.append(Clustering.from_dict(c, api_key=api_key))

    for c in data['classifiers']:
        retlist.append(Classifier.from_dict(c, api_key=api_key))

    return retlist


class BaseModel(object):
    URL_TPL = None
    URL_CREATE_TPL = None

    def __init__(self, name, slug, header, api_key=None, **kwargs):
        self.api_key = api_key
        self.name = name
        self.slug = slug
        self.header = header

    @classmethod
    def from_dict(cls, data, api_key=None):
        c = cls(**data)
        c.api_key = api_key
        return c

    def predict(self, pattern):
        p = ','.join(unicode(p) for p in pattern)

        url = self.URL_TPL % self.slug
        response = http_post(url, self.api_key, p)
        if response.status != 200:
            raise TypeError('Request error: %s' % response.read())
        return json.loads(response.read())


class Classifier(BaseModel):
    URL_TPL = '/api/classifier/%s/predict'

    @classmethod
    def create(cls, api_key, name, dataset):
        content = dataset_to_str(dataset)
        data = dict(name=name, data=content)
        data = urllib.urlencode(data)

        url = '/api/classifier/'
        h = {}
        h['Content-type'] = 'application/x-www-form-urlencoded'
        h['Accept'] = 'text/plain'

        response = http_post(url, api_key, data, h)
        if response.status != 200:
            raise ValueError(response.read())
        data = json.loads(response.read())
        return Classifier.from_dict(data)


class Clustering(BaseModel):
    URL_TPL = '/api/clustering/%s/predict'

    @classmethod
    def create(cls, api_key, name, dataset, clusters):
        content = dataset_to_str(dataset)
        data = dict(name=name, data=content, clusters=clusters)
        data = urllib.urlencode(data)

        url = '/api/grouper/'
        h = {}
        h['Content-type'] = 'application/x-www-form-urlencoded'

        response = http_post(url, api_key, data, h)
        if response.status != 200:
            raise ValueError(response.read())
        data = json.loads(response.read())
        return Clustering.from_dict(data)
