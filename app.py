import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World! This is a minimal test application for AWS App Runner."

if __name__ == '__main__':
    # App Runner sets PORT environment variable
    port = int(os.environ.get('PORT', 8080))
    # Important: host must be 0.0.0.0 for App Runner
    app.run(host='0.0.0.0', port=port, debug=False)
