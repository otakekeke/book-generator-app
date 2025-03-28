import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World! This is a minimal test application for AWS App Runner."

# This is the simplest possible Flask application
# No additional configuration to minimize potential issues
