import subprocess

def run_command(command):
    command = command.split()
    return subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout


print(run_command(""))
