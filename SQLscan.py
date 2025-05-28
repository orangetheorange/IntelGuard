import urllib.parse

def url_to_sqlmap_command(url):
    base_cmd = f"python sqlmap/sqlmap.py -u \"{url}\" --batch --level=3 --risk=2 --dbs --threads=5 --random-agen"
    return base_cmd