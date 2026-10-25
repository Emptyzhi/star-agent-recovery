import json
import socket
import struct


def read_exact(stream, length):
    data = bytearray()
    while len(data) < length:
        chunk = stream.recv(length-len(data))
        if not chunk:
            raise ConnectionError('resident IPC frame ended early')
        data.extend(chunk)
    return bytes(data)


def exchange(endpoint, operation, payload):
    if not endpoint.startswith('unix://'):
        raise ValueError('resident endpoint must be a Unix socket')
    frame = json.dumps(dict(operation=operation, payload=payload)).encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(1200)
        stream.connect(endpoint[7:])
        stream.sendall(struct.pack('!I', len(frame))+frame)
        length = struct.unpack('!I', read_exact(stream, 4))[0]
        response = json.loads(read_exact(stream, length))
    if response['ok'] is not True:
        raise RuntimeError('resident IPC failure: '+response['error'])
    return response['result']
