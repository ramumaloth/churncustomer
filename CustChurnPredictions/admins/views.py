from MySQLdb._mysql import result
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages
import requests
import json
BASE_URL = 'http://localhost:8000/'

# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')
        elif usrid == 'Admin' and pswd == 'Admin':
            return render(request, 'admins/AdminHome.html')
        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})
def AdminHome(request):
    return render(request, 'admins/AdminHome.html')

def ViewRegisteredUsers(request):
    ENDPOINT = "usrreg/"
    response = requests.get(BASE_URL + ENDPOINT)
    code = response.status_code
    resp = response.json()
    #print('Response is code is:', resp)
    return render(request,"admins/ViewRegistredUsers.html",{"data":resp})

def AdminActivaUsers(request):
    if request.method == 'GET':
        id = int(request.GET.get('uid'))
        status = 'activated'
        data = {'id':id,'status':status}
        ENDPOINT = "usrreg/"
        resp = requests.patch(BASE_URL + ENDPOINT+str(id)+'/',headers={'Content-Type': 'application/json'},data=json.dumps(data))
        #print(resp.status_code)
        #print(resp.json())
        return redirect('/ViewRegisteredUsers')

def AdminDeleteUsers(request):
    if request.method == 'GET':
        id = int(request.GET.get('uid'))
        ENDPOINT = "usrreg/"
        resp = requests.delete(BASE_URL + ENDPOINT+str(id)+'/')
        #print(resp.status_code)
        #print(resp.json())
        return redirect('/ViewRegisteredUsers')