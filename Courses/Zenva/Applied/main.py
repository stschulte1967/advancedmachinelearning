# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:19:42 2021

@author: steph
"""

from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello from Flask'
