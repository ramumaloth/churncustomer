import requests
class ProcessDatasets:
    def process(self):
        print('Working')
        BASE_URL = 'http://localhost:8000/'
        ENDPOINT = "df1api/"
        # data = json.dumps(dict_data)
        response = requests.get(BASE_URL + ENDPOINT)
        #code = response.status_code
        #print('Response is code is:', code)
        return response
    def processSecond(self):
        BASE_URL = "http://localhost:8000/"
        ENDPOINT = "df2api/"
        response = requests.get(BASE_URL+ENDPOINT)
        return response
    def processThird(self):
        BASE_URL = "http://localhost:8000/"
        ENDPOINT = "df3api/"
        response = requests.get(BASE_URL+ENDPOINT)
        return response