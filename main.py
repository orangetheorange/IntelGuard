import socket_processes
import scanner_load

if __name__ == "__main__":
    target = "localhost"
    target, iden, stat, open_ports, os = socket_processes.scan(target)
    command1 = scanner_load.process_scan(target, iden, stat, open_ports, os)
    print(command1)


