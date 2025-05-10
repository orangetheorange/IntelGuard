import socket_processes
import scanner_load
import customtkinter

def scan(target):
    target, iden, stat, open_ports, os = socket_processes.scan(target)
    print(f"{target}, {iden}, {stat}, {open_ports}, {os}")
    # scanner_load.generate_command(f"{target}, {iden}, {stat}, {open_ports}, {os}")
