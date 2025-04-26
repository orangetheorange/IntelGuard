def build_sqlmap_command(
    base_url,
    data=None,
    cookies=None,
    method="GET",
    level=None,
    risk=None,
    dbs=False,
    dump=False
):
    """
    Builds a sqlmap command based on input parameters.
    """
    command = ["sqlmap", f"-u \"{base_url}\""]

    if method.upper() == "POST" and data:
        command.append(f"--data=\"{data}\"")

    if cookies:
        command.append(f"--cookie=\"{cookies}\"")

    if level:
        command.append(f"--level={level}")
    if risk:
        command.append(f"--risk={risk}")
    if dbs:
        command.append("--dbs")
    if dump:
        command.append("--dump")

    return " ".join(command)


# Example usage
cmd = build_sqlmap_command(
    base_url="http://example.com/item.php?id=1",
    level=3,
    risk=2,
    dbs=True,
    dump=True
)

print(cmd)
