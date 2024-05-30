import socket
import pickle
from multiprocessing import Process
import argparse
import gym


def start_server(port):
    env = gym.make("CartPole-v1")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', port))
        s.listen()
        print(f'Server listening on port {port}')
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                command = pickle.loads(data)
                response = None

                if command['action'] == 'reset':
                    observation = env.reset()
                    response = observation
                elif command['action'] == 'step':
                    observation, reward, done, info = env.step(command['value'])
                    response = (observation, reward, done, info)
                elif command['action'] == 'close':
                    env.close()
                    response = 'Environment closed'
                    conn.sendall(pickle.dumps(response))
                    break

                if response is not None:
                    conn.sendall(pickle.dumps(response))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a simulation environment server.')
    parser.add_argument('--port', type=int, required=True, help='Port number to listen on')
    args = parser.parse_args()
    start_server(args.port)
