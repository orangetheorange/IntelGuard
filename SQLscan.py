import urllib.parse

def url_to_sqlmap_command(url):
    # Parse the URL
    parsed = urllib.parse.urlparse(url)
    query = parsed.query

    # If no query parameters, cannot scan with sqlmap parameters
    if not query:
        return "URL has no parameters to test with sqlmap."

    # Construct the base sqlmap command
    base_cmd = f"sqlmap -u \"{url}\" --batch --level=3 --risk=2 --dbs --threads=5 --random-agen"

    # Optionally, you can add more sqlmap options here

    return base_cmd
