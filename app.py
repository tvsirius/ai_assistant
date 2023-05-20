from server import server, shutdown_server

from werkzeug.serving import make_server

print('Starting app...')
print('Running on http://127.0.0.1:5000')

if __name__ == '__main__':
    server_w = make_server('localhost', 5000, server)
    # server.run("127.0.0.1", 5000)
    try:
        server_w.serve_forever()
    except KeyboardInterrupt:
        print('Server shut down.')
    server_w.shutdown()
    server_w.server_close()
