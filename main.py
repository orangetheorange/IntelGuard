import socket_processes

if __name__ == "__main__":
    target = "localhost"
    target, iden, stat, open_ports, os = socket_processes.scan(target)


