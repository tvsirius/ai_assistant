# import atexit

from server import server


# atexit.register(server_shutdown)

print('Starting app...')


if __name__ == '__main__':
    server.run("127.0.0.1", 5000)
