# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:12:19 2021

@author: 80686
"""
import pymysql

def mysqltest(x):
    a={}
    conn=pymysql.connect('localhost','root','9705165')
    conn.select_db('hdzz')
    cur=conn.cursor()

    cur.execute("select post from test where id =%d;"%x)
    res=cur.fetchone()
    a.update({'post':res[0]})
    cur.execute("select level from test where id =%d;"%x)
    res=cur.fetchone()
    a.update({'level':res[0]})
    cur.close()
    conn.commit()
    conn.close()
    print('sql执行成功')
    return a

def mysqlapriori():
    conn=pymysql.connect('localhost','root','9705165')
    conn.select_db('hdzz')
    cur=conn.cursor()
    a=[]
    for i in range(8):
        b=[]
        for j in range(1,5):
            cur.execute("select item%d from apriori where id = %d;"%(j,i))
            res=cur.fetchone()
            if res[0] != '':
                b.append(res[0])
        a.append(b)
    cur.close()
    conn.commit()
    conn.close()
    print('sql执行成功')
    return a