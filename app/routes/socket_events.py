from flask_socketio import emit, join_room, leave_room
from flask_socketio import SocketIO

def init_socketio(socketio_instance: SocketIO):
    socketio = socketio_instance

    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        emit('message', {'data': 'Connected to server!'})
        socketio.start_background_task(background_task)
        
    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')

    def background_task():
        while True:
            socketio.emit('update', {'data': 'Periodic update'})
            print('Periodic update sent')
            socketio.sleep(1)