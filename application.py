from flask import Flask,Blueprint,render_template,request,session
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import pymysql
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_percentage_error



model = pd.read_pickle("model/arima_model.model") 
# Define folder to save uploaded files to process further
UPLOAD_FOLDER = os.path.join('static')
 
# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}
 
app = Flask(__name__, template_folder='templates', static_folder='static')
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict")
def show_pred():
    pred=model.predict(steps=30)
    pred=pd.DataFrame(pred).iloc[-30:,:]

    fig, ax = plt.subplots(figsize=(10, 3))
    graph=pred.plot(linewidth=2, label='predicción', ax=ax)
    ax.set_title('Predicción')
    ax.legend()
    fig.savefig("./static/my_plot.png");


    return render_template("predict.html",tables=[pred.to_html(classes="data")],titles=pred.columns.values)

@app.route("/update")
def update():
    return render_template("update.html")

@app.route('/upload',  methods=("POST", "GET"))
def uploadFile():
    try:
        if request.method == 'POST':
            # upload file flask
            uploaded_df = request.files['uploaded-file']
    
            # Extracting uploaded data file name
            data_filename = secure_filename(uploaded_df.filename)
    
            # flask upload file to database (defined uploaded folder in static path)
            uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
    
            # Storing uploaded file path in flask session
            session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)

            uploaded_df=pd.read_csv("static/"+data_filename)

            uploaded_df['Date'] = pd.to_datetime(uploaded_df['Date'])

            engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(user = "admin_cloud", pw = "grupo2DS", host = "database-1.cf0hxwxsba9n.us-east-2.rds.amazonaws.com", db = 'users_web'))
    
            uploaded_df.to_sql(name='usuarios_web', con=engine, if_exists= 'append', index=False)
    
            return render_template('upload.html')

    except:
            return render_template('fail_upload.html')


@app.route("/score", methods=['GET', 'PUT'])
def retrain():
    try:
        username = "admin_cloud"
        password = "grupo2DS"
        host = "database-1.cf0hxwxsba9n.us-east-2.rds.amazonaws.com" 
        port = "3306"

        db = pymysql.connect(host = host,
                     user = username,
                     password = password,
                     cursorclass = pymysql.cursors.DictCursor)
        cursor = db.cursor()
        cursor.connection.commit()
        use_db = ''' USE users_web'''
        cursor.execute(use_db)

        sql = '''SELECT * FROM usuarios_web'''
        cursor.execute(sql)

        mi_lista = cursor.fetchall()
        data = pd.DataFrame(mi_lista)
        test = data.Users[-30:]
        preds = model.predict(30)
        preds=pd.DataFrame(preds).iloc[:-30,:]

        mape =  mean_absolute_percentage_error(test, preds)

        if mape>0.2:

            db.close()
            return render_template('mape_fail.html',mape=mape)

            
        db.close()
        return render_template('mape_win.html',mape=mape)
      

    except:
        return render_template('fail_score.html')

    
@app.route("/fit")
def fit_model(model):
        username = "admin_cloud"
        password = "grupo2DS"
        host = "database-1.cf0hxwxsba9n.us-east-2.rds.amazonaws.com" 
        port = "3306"

        db = pymysql.connect(host = host,
                     user = username,
                     password = password,
                     cursorclass = pymysql.cursors.DictCursor)
        cursor = db.cursor()
        cursor.connection.commit()
        use_db = ''' USE users_web'''
        cursor.execute(use_db)

        sql = '''SELECT * FROM usuarios_web'''
        cursor.execute(sql)

        mi_lista = cursor.fetchall()
        data = pd.DataFrame(mi_lista)
        train = data.Users
        model.fit(train)

        pickle.dump(model, open('model/arima_model.model', 'wb'))
        with open('model/arima_model', "rb") as reentrenado:
                model = pickle.load(reentrenado)

        db.close()
        return render_template('fit_model.html')
if __name__=='__main__':
    app.run(debug = True)