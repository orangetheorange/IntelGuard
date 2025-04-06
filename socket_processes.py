import os
import platform
import re
import socket
import subprocess
import sys


def grant_sudo_access():
    if subprocess.getoutput('id -u') != '0':  # Check if not running as root
        subprocess.run(['pkexec', sys.executable] + sys.argv)  # Request sudo access
        sys.exit(0)  # Exit after sudo request


def ping(host):
    param = "-n" if platform.system().lower() == "windows" else "-c"
    command = ["ping", param, "4", host]
    response = os.system(" ".join(command))
    return "up" if response == 0 else "down"


def portscan(host, args="-sV -O"):
    process = subprocess.Popen(["sudo", "nmap"] + args.split() + [host], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, bufsize=1)
    port_list = []
    os_info = ""

    for line in iter(process.stdout.readline, ''):
        if "open" in line:
            port_list.append(line.strip())
        if "OS details" in line:
            os_info = line.strip().replace("OS details:", "").strip()
    process.wait()

    ports = {}
    for entry in port_list:
        match = re.search(r"(\d+)/tcp\s+open\s+\S+\s+(.+)", entry)
        if match:
            ports[match.group(1)] = match.group(2)

    return ports, os_info


def identify_input(input_string):
    domain_regex = r'^[a-zA-Z0-9.-]+\.[a-zAZ0-9]{2,}$'
    url_regex = r'^(https?://)?([a-zA-Z0-9.-]+)(/.*)?$'
    ip_regex = r'^(\d{1,3}\.){3}\d{1,3}$'
    if re.match(ip_regex, input_string):
        try:
            socket.inet_pton(socket.AF_INET, input_string)
            return "IP address"
        except socket.error:
            return "Invalid IP address"
    elif re.match(domain_regex, input_string):
        return "Domain name"
    elif re.match(url_regex, input_string):
        return "URL"
    return "Unknown format"


def scan(target):
    print("----- Information Gathering -----")
    grant_sudo_access()  # Prompt for sudo password before running nmap
    iden = identify_input(target)
    print("Target input type:", iden)
    stat = ping(target)
    print("The host is", stat)
    open_ports, os_info = portscan(target)
    print("Open Ports:")
    for x in open_ports:
        print(x, open_ports[x])
    print("OS Details:", os_info if os_info else "OS details not detected")

    # Return the result as a dictionary instead of saving to a file
    result = {
        "Target": target,
        "Iden": iden,
        "Stat": stat,
        "Open Ports": open_ports,
        "OS": os_info if os_info else "OS details not detected"
    }

    return result
