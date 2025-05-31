import subprocess
import socket_processes
from urllib.parse import urlparse, parse_qs
import SQLscan
import report
from tkinter import messagebox
import commandgen

def has_parameters(url: str) -> bool:
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    return len(query_params) > 0

def run_command(command):
    command = command.split()
    return subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout

def scanTar(target):
    reportinp = ""
    result = socket_processes.scan(target)
    if result.get("Stat") == "down":
        messagebox.showwarning("Scan Failed", "The target host is currently down or does not exist, please try again later.")
        return f"The target host {target["Target"]} is currently down or does not exist, please double check for any typos, misinformation or try again later."
    if result.get("Iden") == "url":
        if has_parameters(target):
            cmd = SQLscan.url_to_sqlmap_command(target)
            if cmd:
                try:
                    # Run sqlmap command and capture output
                    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    sqlmap_output = completed.stdout
                    print(sqlmap_output)
                    reportinp += f"\n--- SQLMap scan results for {target} ---\n"
                    reportinp += sqlmap_output
                except subprocess.CalledProcessError as e:
                    reportinp += f"\n[Error] SQLMap scan failed for {target}:\n{e.stderr}\n"
            else:
                reportinp += f"\n[Info] No parameters found in URL to scan: {target}\n"
        else:
            reportinp += f"\n[Info] URL has no parameters to scan: {target}\n"
        target = urlparse(target).hostname
    else:
        reportinp += f"\n[Info] Target is not a URL: {target}\n"
    result["Target"] = target
    cmd = commandgen.command(result)
    vulnResult = run_command(cmd)
    rep = report.report(vulnResult)
    print(rep)
    return rep
