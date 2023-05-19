
from server import server, shutdown_server


from werkzeug.serving import make_server

import signal
def shutdown(signal, frame):
    print('Shutting down server...')
    shutdown_server()


print('Starting app...')
print('Running on http://127.0.0.1:5000')


if __name__ == '__main__':
    signal.signal(signal.SIGINT, shutdown)
    server_w = make_server('localhost', 5000, server)
    try:
        server_w.serve_forever()
    except KeyboardInterrupt:
        server_w.shutdown()
        server_w.server_close()
        print('Server shut down.')
    # server.run("127.0.0.1", 5000)
