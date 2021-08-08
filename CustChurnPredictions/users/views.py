from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from users.forms import UserRegistrationForm
import requests
from django.http import JsonResponse
from api.churnapi.models import UserRegistrationModel
import pandas as pd
from .utilities.FirstDatasetProcess import ProcessDatasets

BASE_URL = 'http://localhost:8000/'
# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            data = form.cleaned_data
            ENDPOINT = "usrreg/"
            #data = json.dumps(dict_data)
            response = requests.post(BASE_URL + ENDPOINT, json=data)
            code = response.status_code
            print('Response is code is:', code)
            if code==201:
                messages.success(request, 'Registration Success')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)

        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')

    return render(request, 'UserLogin.html', {})

def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def FirstDatasetAction(request):
    obj1 = ProcessDatasets()
    result = obj1.process()
    result = result.json()
    df = result['dataset']
    df = pd.DataFrame(df)
    #print(df.columns)
    df = df[['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','OnlineBackup','DeviceProtection','TechSupport','TotalCharges','Churn']]
    df = df.head(200)
    df = df.to_html
    dict_lg = result['dict_lg']
    dict_dt = result['dict_dt']
    dict_rf = result['dict_rf']
    dict_ada = result['dict_ada']
    dict_mlp = result['dict_mlp']

    return render(request,'users/datasetoneresult.html',{'df':df,'dict_lg':dict_lg,'dict_dt':dict_dt,'dict_rf':dict_rf,'dict_ada':dict_ada,'dict_mlp':dict_mlp})

def SecondDatasetAction(request):
    obj2 = ProcessDatasets()
    rslt = obj2.processSecond()
    result = rslt.json()
    df = result['dataset']
    df = pd.DataFrame(df)
    # print(df.columns)
    df = df[['state', 'voice_mail_plan', 'total_day_minutes', 'total_day_calls', 'total_night_charge', 'total_eve_charge']]
    df = df.head(200)
    df = df.to_html
    dict_lg = result['dict_lg']
    dict_dt = result['dict_dt']
    dict_rf = result['dict_rf']
    dict_ada = result['dict_ada']
    dict_mlp = result['dict_mlp']

    return render(request, 'users/datasetworeresult.html',
                  {'df': df, 'dict_lg': dict_lg, 'dict_dt': dict_dt, 'dict_rf': dict_rf, 'dict_ada': dict_ada,
                   'dict_mlp': dict_mlp})

def ThirdDatasetAction(request):
    obj2 = ProcessDatasets()
    rslt = obj2.processThird()
    result = rslt.json()
    df = result['dataset']
    df = pd.DataFrame(df)
    # print(df.columns)
    df = df[
        ['CustomerID', 'MonthlyRevenue', 'DirectorAssistedCalls', 'RoamingCalls', 'UnansweredCalls', 'HandsetPrice']]
    df = df.head(200)
    df = df.to_html
    dict_lg = result['dict_lg']
    dict_dt = result['dict_dt']
    dict_rf = result['dict_rf']
    dict_ada = result['dict_ada']
    dict_mlp = result['dict_mlp']

    return render(request, 'users/datasethreereresult.html',
                  {'df': df, 'dict_lg': dict_lg, 'dict_dt': dict_dt, 'dict_rf': dict_rf, 'dict_ada': dict_ada,
                   'dict_mlp': dict_mlp})
