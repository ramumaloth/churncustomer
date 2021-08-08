import requests
import json

BASE_URl = 'http://127.0.0.1:8000/'
ENDPOINT= 'apijsoncbv'

#resp = requests.get(BASE_URl+ENDPOINT)
#dict = resp.json()
#print('Data from Django Applicati0n')
#print('#'*50)
#print('Employee Number :',dict['eno'])
#print('Employee Name :',dict['ename'])
#print('Employee Salary :',dict['esal'])
#print('Employee Address :',dict['eaddr'])

#print(dict)

BASE_URl = 'http://127.0.0.1:8000/'
ENDPOINT= 'usrreg/'


def get_resource():
    url = BASE_URl+ENDPOINT
    data = {'loginid':'alex','password':'Alex@141'}
    print('URL is ',url)
    #resp = requests.get(url,data = json.dumps(data))
    resp = requests.get(url)
    print(resp.status_code)
    print(resp.json())
get_resource()

'''
def get_all():
    url = BASE_URl+ENDPOINT
    print('URL is ',url)
    resp = requests.get(url)
    print(resp.json())
#get_all()


def create_resource():
    new_emp = {
    'eno':1003,
    'ename':'Sai Mummy',
    'esal':2000,
    'eaddr':'East AnandBagh'
    }
    resp = requests.post(BASE_URl+ENDPOINT,data = json.dumps(new_emp))
    print(resp.status_code)
    print(resp.json())

#create_resource()

def update_resource(id):
    new_emp = {
    'esal':56000,
    'eaddr':'naivy Mumbai'
    }
    resp = requests.put(BASE_URl+ENDPOINT+str(id)+'/',data = json.dumps(new_emp))
    print(resp.status_code)
    print(resp.json())
#update_resource(1)
#get_resource()

def delete_resource(id):
    resp = requests.delete(BASE_URl+ENDPOINT+str(id)+'/')
    print(resp.status_code)
    print(resp.json())
delete_resource(1)
'''