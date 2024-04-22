from django.shortcuts import render,redirect
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage
import pymysql
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.layers import Dense, Dropout
# import tensorflow as tf
# tf.compat.v1.losses.sparse_softmax_cross_entropy
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import keras
from .forms import FeedbackForm
import keras.layers
from sklearn.ensemble import RandomForestClassifier
from django.contrib.auth.models import User
from django.contrib.auth import login
from django.conf import settings
from datetime import datetime
from django.contrib import messages
#from .utils import send_email,append_results_to_excel
from email.mime.text import MIMEText
import smtplib
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from datetime import datetime
from .models import FilesUpload
from django.core.files.storage import FileSystemStorage 
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#imported libraries

def write_to_google_sheets(data,label):
    # Use the JSON file downloaded during the Google Sheets API setup
    creds = None
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    # The ID and range of the spreadsheet.
    SPREADSHEET_ID = '1cmT7qhcKOLi6b2--7OC0BtPhwXsgiLMD0gfUnol7_Jo'
    RANGE_NAME = 'Sheet1' #sheet number
    # Load credentials
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        'creds.json', SCOPES) #get your credentials for the google spread sheets api
    # Connect to the Google Sheet
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SPREADSHEET_ID).sheet1 #sheet link
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SPREADSHEET_ID).sheet1
    current_timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')
    row_data = [current_timestamp] + [data['timestamp']] + data['values'] + [label]
    sheet.append_row(row_data)

def send_email_with_link(sheet_link, recipient_email):
    # Your email credentials
    email_address = 'shaikzohaibpardeep@gmail.com'  #sender email address.
    # Replace 'your_app_password' with the App Password generated.
    email_password = 'vqry gqvr gfdo pode'#app password for the selected email address.

    # Set up the email
    subject = 'Water Quality Report'
    body = f'Here is the link to the Google Sheet: {sheet_link}'
    message = MIMEText(body)
    message['Subject'] = subject #subject
    message['From'] = email_address 
    message['To'] = recipient_email

    # Connect to the SMTP server and send the email
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(email_address, email_password)
        server.sendmail(email_address, recipient_email, message.as_string())

#plots generation
def generate_plot(data):
    if isinstance(data, pd.DataFrame):
        label = data.groupby('labels').size()
    else:
        raise ValueError("Unsupported data type. Expected pandas DataFrame.")

    label.plot(kind="bar")
    plt.title("Water Quality Graph, 0 (Good quality) & 1 (Poor Quality)")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    fn = f"C:/Users/shaik/Desktop/Projects Workspace/WaterQuality/WaterQuality/WaterqualityApp/static/images/plots/plot_{timestamp}.png"
    plt.savefig(fn)
    plt.close()  # Close the plot to release resources

#global variables declaration
X, Y, dataset, X_train, X_test, y_train, y_test = None, None, None, None, None, None, None
algorithms, accuracy, fscore, precision, recall, classifier = [], [], [], [], [], None
    
def ProcessData(request):
    if request.method == 'POST':
        # render_part = True
        # part = {
        #     render_part: render_part,
        # }
        global X, Y, dataset, X_train, X_test, y_train, y_test
        file2 = request.FILES["file"]
        print(request.user)
        document = FilesUpload.objects.create(file=file2)
        document.save()
        print(file2.name)
        dataset = pd.read_csv("media/uploads/"+file2.name)
        dataset.fillna(0, inplace=True)
        generate_plot(dataset)
        columns = dataset.columns
        temp = dataset.values
        dataset = dataset.values
        X = dataset[:, 2:dataset.shape[1] - 1]
        Y = dataset[:, dataset.shape[1] - 1]
        Y = Y.astype(int)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        output = '<center><table  class="table table-bordered table-sm" border=1 align=center position=center width=100%>'
        font = '<font size="" color="black">'
        output += '<thead class="thead-active">'
        output += "<tr>"
        for i in range(len(columns)):
            output += "<th class='tableheader'>" + font + columns[i] + "</th>"
        output += "</tr>"
        for i in range(len(temp)):
            output += "<tr class='tabledata'>"
            for j in range(0, temp.shape[1]):
                output += '<td><font size="" color="black">' + str(temp[i, j]) + '</td>'
            output += "</tr>"
        output += "</table></center>"
        context = {'data': output,'part': True } 
        return render(request, 'UserScreen.html', context)
    else:
        return render(request, 'UserScreen.html', {})
    
def TrainRF(request):
    global X, Y
    global algorithms, accuracy, fscore, precision, recall, classifier
    if request.method == 'GET':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        cls = RandomForestClassifier()
        cls.fit(X, Y)
        classifier = cls
        predict = cls.predict(X_test)
        p = precision_score(y_test, predict,average='macro') * 100
        r = recall_score(y_test, predict,average='macro') * 100
        f = f1_score(y_test, predict,average='macro') * 100
        a = accuracy_score(y_test,predict)*100
        algorithms.append("Random Forest")
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        output = '<table class = "table table-bordered table-sm" border=1 align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th class='tableheader'>"+font+arr[i]+"</th>"
        output += "</tr>"
        for i in range(len(algorithms)):
            output +="<tr class='tabledata'><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td><td>"+font+str(fscore[i])+"</td></tr>"
        output += "</table>"
        context= {'data': output,'part': False }
        return render(request, 'UserScreen.html', context)

def TrainLSTM(request):
    if request.method == 'GET':
        global X, Y, algorithms, accuracy, fscore, precision, recall
        X1 = np.reshape(X, (X.shape[0], X.shape[1], 1))
        Y1 = to_categorical(Y)
        X1 = X1.astype(np.float32)
        Y1 = Y1.astype(np.float32)
        #print(X1.shape)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2)
        if request.method == 'GET':
            lstm_model = Sequential()
            lstm_model.add(keras.layers.LSTM(100, input_shape=(X_train.shape[1], 1)))
            #lstm_model.add(keras.layers.LSTM(100,input_shape=(X_train.shape[1], X_train.shape[2])))
            lstm_model.add(Dropout(0.5))
            lstm_model.add(Dense(100, activation='relu'))
            lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
            lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            lstm_model.fit(X1, Y1, epochs=5, batch_size=32, validation_data=(X_test, y_test))             
            print(lstm_model.summary())#printing model summary
            predict = lstm_model.predict(X_test)
            predict = np.argmax(predict, axis=1)
            testY = np.argmax(y_test, axis=1)
            p = precision_score(testY, predict,average='macro',zero_division=1) * 100
            r = recall_score(testY, predict,average='macro') * 100
            f = f1_score(testY, predict,average='macro') * 100
            a = accuracy_score(testY,predict)*100
            algorithms.append("LSTM")
            accuracy.append(a)
            precision.append(p)
            recall.append(r)
            fscore.append(f)
            arr = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
            output = '<table class="table table-bordered table-sm" border=1 align=center width=100%>'
            font = '<font size="" color="black">'
            output += "<tr>"
            for i in range(len(arr)):
                output += "<th class='tableheader'>"+font+arr[i]+"</th>"
            output += "</tr>"
            for i in range(len(algorithms)):
                output +="<tr class='tabledata'><td>"+font+str(algorithms[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td><td>"+font+str(fscore[i])+"</td></tr>"
            context= {'data': output,'part': False }
            return render(request, 'UserScreen.html', context)
        
def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def PredictAction(request):
    if request.method == 'POST':
        global classifier
        file3 = request.FILES["file"]
        document = FilesUpload.objects.create(file=file3)
        document.save()
        print(file3.name)
        test = pd.read_csv("media/uploads/"+file3.name)
        
        # con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'Waterquality',charset='utf8')
        db_connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='Waterquality', charset='utf8')
        db_cursor = db_connection.cursor()

        # Modify the query to fetch email for the current user
        student_sql_query = "SELECT email FROM signup WHERE username = %s;"
        res = db_cursor.execute(student_sql_query, (str(request.user),))

        if res:
            email = db_cursor.fetchone()[0]
            print("Email found:", email)
        else:
            print("Email not found for user:", request.user)

        test.fillna(0, inplace=True)
        test = test.values
        X = test[:, 2:dataset.shape[1] - 1]
        predict = classifier.predict(X)   
        labels = ['Good Quality', 'Poor Quality'] 
        arr = ['Test Data', 'Water Quality Forecasting Result']
        for i in range(min(len(predict), len(test))):
            data = {
                'timestamp': str(test[i][0]),
                'values': [str(val) for val in test[i][1:]],
            }
            label = labels[predict[i]]
            write_to_google_sheets(data, label)
        sheet_link = f'https://docs.google.com/spreadsheets/d/1cmT7qhcKOLi6b2--7OC0BtPhwXsgiLMD0gfUnol7_Jo'

        # Send an email with the link to the recipient
        send_email_with_link(sheet_link,email)   #recipient email address         
        output = '<table  class="table table-bordered table-sm" align=center width=100%>'
        font = '<font size="" color="black">'
        output += "<tr>"
        for i in range(len(arr)):
            output += "<th class='tableheader'>" + font + arr[i] + "</th>"
        output += "</tr>"
        for i in range(len(predict)):
            output +="<tr class='tabledata'><td>"+font+str(test[i])+"</td><td>"+font+str(labels[predict[i]])+"</td></tr>"
            #print(str(test[i])+" "+str(labels[predict[i]]))
        messages.success(request, 'File Uploaded & Mail Sent successfully!')
        context = {'data': output, 'part': True}    
        return render(request, 'UserScreen.html', context)

def UserLogin(request):
    if request.method == 'GET':
        return render(request, 'UserLogin.html', {})  
    else:
        return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})

def Signup(request):
    if request.method == 'GET':
        return render(request, 'Signup.html', {})

def UserLoginAction(request):
    global uname
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        print(username, password)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'Waterquality',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            user=User.objects.get(username=username, password=password)
            if user:
                login(request, user)

            context= {'data':'welcome '+uname,'part': True}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed. Please retry','part': True}
            return render(request, 'UserLogin.html', context)        

def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('name', "")
        password = request.POST.get('pass', "")
        contact = request.POST.get('conn', "")
        #gender = request.POST.get('t4', False)
        global email
        email = request.POST.get('email', "")
        address = request.POST.get('add', "")
        print(username, password, contact, email, address)
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'Waterquality',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break
        if output == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'Waterquality',charset='utf8')
            db_cursor = db_connection.cursor()
            print(username, password, contact, email, address)
            student_sql_query = "INSERT INTO signup(username,password,contact_no,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            user = User(username=username, password=password)
            user.save()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = 'Signup Process Successful. Please redirect to Login'
        context= {'data':output}
        return render(request, 'Signup.html', context)
#My code
def Feedback(request):
    if request.method == 'POST':
        print(request.user)
        Feedback=request.POST['Feedback']
        #print(Feedback)
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #print(str(current_timestamp))
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'Waterquality',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO feedback2 VALUES (\""+str(request.user.id)+"\",\""+str(Feedback)+"\",\""+str(current_timestamp)+"\");"
        #print(student_sql_query)
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        #print("user id",request.user.id)
        #print("feedback",Feedback)
        #print("timestamp",current_timestamp)
        #print(db_cursor.rowcount, "Record Inserted")
        messages.success(request, 'Feedback Submitted successfully')

    return render(request, 'Feedback.html')

