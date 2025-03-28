import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World! This is a minimal test application for AWS App Runner."

# App Runner expects the application to be available at port 8080
port = int(os.environ.get('PORT', 8080))

# For App Runner, we need to use the application directly
# Do not use the if __name__ == '__main__' block
# App Runner will call the 'app' object directly
if __name__ == '__main__':
    # This is only for local testing
    app.run(host='0.0.0.0', port=port, debug=False)
