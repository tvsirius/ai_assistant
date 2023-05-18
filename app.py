
from server import server

from werkzeug.serving import make_server



print('Starting app...')


if __name__ == '__main__':
    server_w = make_server('localhost', 5000, server)
    server_w.serve_forever()
    # server.run("127.0.0.1", 5000)
