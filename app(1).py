# -*- coding: UTF-8 -*- 
from flask import Flask, jsonify
from flask_cors import CORS
from flask import Flask, request
import threading
import time
import pymysql
from ariori import loadDataSet,apriori,draw,CAL
from mysql import mysqltest,mysqlapriori

# configurations
DEBUG = False

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

# 读取的数据文件的路径
DATA_FILE_PATH = '/home/nash5/Desktop/class/data/pitchOnly.csv'
jobs = [{}]

# 用于srcData和resData的锁
lock = threading.Lock()

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/testapipost', methods=['post'])
def test():
    data = request.get_json(silent=True)
    print(data) #123
    x = data['firstName']
    print(x)
    x = int(x)
    a=mysqltest(x)
    print(x)
    jobs[0]=a    
    print (jobs)
    return jsonify({'jobs':jobs})
    
@app.route('/apriori', methods=['post'])
def apriori():    
    dataSet=mysqlapriori()
    L, supportData = apriori(dataSet,minSupport=0.2)
    draw(L[1],supportData)
    CAL(L,supportData,minConfidence=0.5)
    return jsonify({'jobs':jobs})




if __name__ == '__main__':

    # 运行flask app
    app.run(host='127.0.0.1',port=5000) #debug=True
