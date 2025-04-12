import csv
import random
import re
import socket
from pathlib import Path
import os

# get the script's directory
script_path = Path(__file__).resolve()

# get the parent directory
parent_dir = script_path.parent

# change current working directory
os.chdir(parent_dir)
print("working directory:", parent_dir)

# function to generate random ipv4 addresses
def generate_random_ip():
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

# function to generate random domain names
def generate_random_domain():
    # generate a random domain name like 'abcxyz.com'
    length = random.randint(5, 10)
    domain = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))
    return domain + ".com"

# function to randomly choose a target: ip address or domain name
def generate_random_target():
    return random.choice([generate_random_ip(), generate_random_domain()])

# function to identify input type based on regex
def identify_input(input_string):
    # regex patterns in lowercase
    domain_regex = r'^[a-zA-Z0-9.-]+\.[a-zA-Z0-9]{2,}$'
    url_regex = r'^(https?://)?([a-zA-Z0-9.-]+)(/.*)?$'
    ip_regex = r'^(\d{1,3}\.){3}\d{1,3}$'
    if re.match(ip_regex, input_string):
        try:
            socket.inet_pton(socket.AF_INET, input_string)
            return "ip address"
        except socket.error:
            return "invalid ip address"
    elif re.match(domain_regex, input_string):
        return "domain name"
    elif re.match(url_regex, input_string):
        return "url"
    return "unknown format"

# mapping ports to their corresponding services
port_service_map = {
    21: "vsftpd 3.0.5",
    22: "openssh 8.9p1",
    80: "apache httpd 2.4.62 ((debian))",
    139: "samba smbd 4.6.2",
    443: "apache httpd 2.4.62 ((debian)) tls",
    445: "samba smbd 4.6.2",
    631: "cups 2.4",
    3306: "mysql 5.7",
    8080: "apache tomcat 9.0.12"
}

# available os options
os_options = ["linux", "windows", "unix", "macos", "freebsd"]

# available status options
status_options = ["up", "down"]

# mapping ports to nmap scripts
port_scripts = {
    21: "ftp-vuln*",
    22: "ssh-vuln*",
    80: "http-vuln*",
    139: "smb-vuln*",
    443: "ssl*",
    445: "smb-vuln*",
    631: "cups-info",
    3306: "mysql-vuln*",
    8080: "http-vuln*",
}

# function to build a custom nmap command based on target and ports
def build_custom_command(target, ports):
    scripts = set()
    for p in ports:
        if p in port_scripts:
            scripts.add(port_scripts[p])
    port_string = ",".join(str(p) for p in sorted(ports))
    if scripts:
        script_string = ",".join(scripts)
        return f"nmap -p {port_string} --script {script_string} {target}"
    else:
        return f"nmap -p {port_string} {target}"

# define the number of rows for the dataset
num_rows = 100000
filename = f'generated_dataset_{num_rows}_lines.csv'

with open(filename, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # write header row
    writer.writerow(["Target", "Iden", "Stat", "Open Ports", "OS", "Custom Command"])
    
    for i in range(num_rows):
        # generate a target, ip or domain
        target = generate_random_target()
        # identify target type using the provided function
        iden = identify_input(target)
        # choose a random status
        stat = random.choice(status_options)
        # choose a random subset of ports (1 to 5)
        n_ports = random.randint(1, 5)
        chosen_ports = random.sample(list(port_service_map.keys()), n_ports)
        # build open ports string like: "'80': 'apache httpd 2.4.62 ((debian))'; '443': 'apache httpd 2.4.62 ((debian)) tls'"
        open_ports_str = "; ".join(f"'{p}': '{port_service_map[p]}'" for p in sorted(chosen_ports))
        # choose a random os option
        os_choice = random.choice(os_options)
        # build custom command string
        custom_cmd = build_custom_command(target, chosen_ports)
        
        writer.writerow([target, iden, stat, open_ports_str, os_choice, custom_cmd])

print(f"generated {num_rows} lines in '{filename}'.")
