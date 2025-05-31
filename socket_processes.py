import os
import platform
import re
import socket
import subprocess
import sys
import shutil
from urllib.parse import urlparse

def ensure_nmap_in_path():
    """Ensure nmap is accessible from the current Python environment on Windows."""
    if platform.system() == "Windows":
        if shutil.which("nmap") is None:
            nmap_path = r"C:\Program Files (x86)\Nmap"
            if os.path.exists(os.path.join(nmap_path, "nmap.exe")):
                os.environ["PATH"] += os.pathsep + nmap_path
            else:
                print("❌ Error: Nmap not found at expected location.")
                return False
        return True
    return True


def grant_sudo_access():
    if platform.system() != "Windows":
        if subprocess.getoutput('id -u') != '0':
            subprocess.run(['pkexec', sys.executable] + sys.argv)
            sys.exit(0)


def ping(host):
    param = "-n" if platform.system().lower() == "windows" else "-c"
    command = ["ping", param, "4", host]
    response = subprocess.getoutput(" ".join(command))
    print(response)

    # Check for signs of successful responses
    if platform.system().lower() == "windows":
        # Handles both English and Chinese responses
        if "TTL=" in response or "ttl=" in response or "的回复" in response or "Reply from" in response:
            return "up"
    else:
        if "bytes from" in response:
            return "up"
    return "down"



def portscan(host, args="-sV -O"):
    if not ensure_nmap_in_path():
        print("❌ Error: nmap is not installed or not in PATH.")
        return {}, ""

    try:
        command = ["nmap"] + args.split() + [host]
        result = subprocess.getoutput(" ".join(command))

        ports = {}
        os_info = ""

        for line in result.splitlines():
            if "/tcp" in line and "open" in line:
                match = re.search(r"(\d+)/tcp\s+open\s+\S+\s+(.+)", line)
                if match:
                    ports[match.group(1)] = match.group(2)
            if "OS details" in line:
                os_info = line.strip().replace("OS details:", "").strip()

        return ports, os_info
    except Exception as e:
        print("❌ Error during portscan:", e)
        return {}, ""


def identify_input(input_string):
    domain_regex = r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
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
    if platform.system() == "Linux":
        grant_sudo_access()

    iden = identify_input(target)
    print("Target input type:", iden)

    # Extract domain from URL if needed
    if iden == "URL":
        parsed = urlparse(target)
        target_host = parsed.hostname or parsed.path.split("/")[0]
    else:
        target_host = target

    stat = ping(target_host)
    print("The host is", stat)

    open_ports, os_info = portscan(target_host)
    print("Open Ports:")
    for port, desc in open_ports.items():
        print(f"{port}: {desc}")

    print("OS Details:", os_info if os_info else "OS details not detected")

    result = {
        "Target": target,
        "Iden": iden,
        "Stat": stat,
        "Open Ports": open_ports,
        "OS": os_info if os_info else "windows"
    }
    return result
