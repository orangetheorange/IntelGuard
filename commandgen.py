service_to_scripts = {
    'ssh': [
        'ssh2-enum-algos.nse', 'ssh-auth-methods.nse', 'ssh-brute.nse',
        'ssh-hostkey.nse', 'ssh-publickey-acceptance.nse', 'sshv1.nse'
    ],
    'http': [
        'http-adobe-coldfusion-apsa1301.nse', 'http-affiliate-id.nse',
        'http-apache-negotiation.nse', 'http-apache-server-status.nse',
        'http-aspnet-debug.nse', 'http-auth.nse', 'http-auth-finder.nse',
        'http-avaya-ipoffice-users.nse', 'http-awstatstotals-exec.nse',
        'http-axis2-dir-traversal.nse', 'http-backup-finder.nse',
        'http-barracuda-dir-traversal.nse', 'http-bigip-cookie.nse',
        'http-brute.nse', 'http-cakephp-version.nse', 'http-chrono.nse',
        'http-cisco-anyconnect.nse', 'http-coldfusion-subzero.nse',
        'http-comments-displayer.nse', 'http-config-backup.nse',
        'http-cookie-flags.nse', 'http-cors.nse', 'http-cross-domain-policy.nse',
        'http-csrf.nse', 'http-date.nse', 'http-default-accounts.nse',
        'http-devframework.nse', 'http-dlink-backdoor.nse', 'http-dombased-xss.nse',
        'http-domino-enum-passwords.nse', 'http-drupal-enum.nse',
        'http-drupal-enum-users.nse', 'http-enum.nse', 'http-errors.nse',
        'http-exif-spider.nse', 'http-favicon.nse', 'http-feed.nse',
        'http-fetch.nse', 'http-fileupload-exploiter.nse', 'http-form-brute.nse',
        'http-form-fuzzer.nse', 'http-frontpage-login.nse', 'http-generator.nse',
        'http-git.nse', 'http-gitweb-projects-enum.nse', 'http-google-malware.nse',
        'http-grep.nse', 'http-headers.nse', 'http-hp-ilo-info.nse',
        'http-huawei-hg5xx-vuln.nse', 'http-icloud-findmyiphone.nse',
        'http-icloud-sendmsg.nse', 'http-iis-short-name-brute.nse',
        'http-iis-webdav-vuln.nse', 'http-internal-ip-disclosure.nse',
        'http-joomla-brute.nse', 'http-jsonp-detection.nse',
        'http-litespeed-sourcecode-download.nse', 'http-ls.nse',
        'http-majordomo2-dir-traversal.nse', 'http-malware-host.nse',
        'http-mcmp.nse', 'http-methods.nse', 'http-method-tamper.nse',
        'http-mobileversion-checker.nse', 'http-ntlm-info.nse',
        'http-open-proxy.nse', 'http-open-redirect.nse', 'http-passwd.nse',
        'http-phpmyadmin-dir-traversal.nse', 'http-phpself-xss.nse',
        'http-php-version.nse', 'http-proxy-brute.nse', 'http-put.nse',
        'http-qnap-nas-info.nse', 'http-referer-checker.nse', 'http-rfi-spider.nse',
        'http-robots.txt.nse', 'http-robtex-reverse-ip.nse', 'http-robtex-shared-ns.nse',
        'http-sap-netweaver-leak.nse', 'http-security-headers.nse', 'http-server-header.nse',
        'http-shellshock.nse', 'http-sitemap-generator.nse', 'http-slowloris.nse',
        'http-slowloris-check.nse', 'http-sql-injection.nse', 'http-stored-xss.nse',
        'http-svn-enum.nse', 'http-svn-info.nse', 'http-title.nse',
        'http-tplink-dir-traversal.nse', 'http-trace.nse', 'http-traceroute.nse',
        'http-trane-info.nse', 'http-unsafe-output-escaping.nse', 'http-useragent-tester.nse',
        'http-userdir-enum.nse', 'http-vhosts.nse', 'http-virustotal.nse',
        'http-vlcstreamer-ls.nse', 'http-vmware-path-vuln.nse', 'http-vuln-*.nse'
    ],
    'ftp': [
        'ftp-anon.nse', 'ftp-bounce.nse', 'ftp-brute.nse',
        'ftp-libopie.nse', 'ftp-proftpd-backdoor.nse', 'ftp-syst.nse',
        'ftp-vsftpd-backdoor.nse', 'ftp-vuln-cve2010-4221.nse'
    ],
    'smtp': [
        'smtp-brute.nse', 'smtp-commands.nse', 'smtp-enum-users.nse',
        'smtp-ntlm-info.nse', 'smtp-open-relay.nse', 'smtp-strangeport.nse',
        'smtp-vuln-cve2010-4344.nse', 'smtp-vuln-cve2011-1720.nse',
        'smtp-vuln-cve2011-1764.nse'
    ],
    'dns': [
        'dns-blacklist.nse', 'dns-brute.nse', 'dns-cache-snoop.nse',
        'dns-check-zone.nse', 'dns-client-subnet-scan.nse', 'dns-fuzz.nse',
        'dns-ip6-arpa-scan.nse', 'dns-nsec3-enum.nse', 'dns-nsec-enum.nse',
        'dns-nsid.nse', 'dns-random-srcport.nse', 'dns-random-txid.nse',
        'dns-recursion.nse', 'dns-service-discovery.nse', 'dns-srv-enum.nse',
        'dns-update.nse', 'dns-zeustracker.nse', 'dns-zone-transfer.nse'
    ],
    'smb': [
        'smb2-capabilities.nse', 'smb2-security-mode.nse', 'smb2-time.nse',
        'smb2-vuln-uptime.nse', 'smb-brute.nse', 'smb-double-pulsar-backdoor.nse',
        'smb-enum-domains.nse', 'smb-enum-groups.nse', 'smb-enum-services.nse',
        'smb-enum-sessions.nse', 'smb-enum-shares.nse', 'smb-enum-users.nse',
        'smb-flood.nse', 'smb-ls.nse', 'smb-mbenum.nse', 'smb-os-discovery.nse',
        'smb-print-text.nse', 'smb-protocols.nse', 'smb-psexec.nse',
        'smb-security-mode.nse', 'smb-server-stats.nse', 'smb-system-info.nse',
        'smb-vuln-*.nse'
    ],
    'mysql': [
        'mysql-audit.nse', 'mysql-brute.nse', 'mysql-databases.nse',
        'mysql-dump-hashes.nse', 'mysql-empty-password.nse', 'mysql-enum.nse',
        'mysql-info.nse', 'mysql-query.nse', 'mysql-users.nse',
        'mysql-variables.nse', 'mysql-vuln-cve2012-2122.nse'
    ],
    'ms-sql': [
        'ms-sql-brute.nse', 'ms-sql-config.nse', 'ms-sql-dac.nse',
        'ms-sql-dump-hashes.nse', 'ms-sql-empty-password.nse', 'ms-sql-hasdbaccess.nse',
        'ms-sql-info.nse', 'ms-sql-ntlm-info.nse', 'ms-sql-query.nse',
        'ms-sql-tables.nse', 'ms-sql-xp-cmdshell.nse'
    ],
    'rdp': [
        'rdp-enum-encryption.nse', 'rdp-ntlm-info.nse', 'rdp-vuln-ms12-020.nse'
    ],
    'snmp': [
        'snmp-brute.nse', 'snmp-hh3c-logins.nse', 'snmp-info.nse',
        'snmp-interfaces.nse', 'snmp-ios-config.nse', 'snmp-netstat.nse',
        'snmp-processes.nse', 'snmp-sysdescr.nse', 'snmp-win32-services.nse',
        'snmp-win32-shares.nse', 'snmp-win32-software.nse', 'snmp-win32-users.nse'
    ],
    'ssl': [
        'ssl-ccs-injection.nse', 'ssl-cert.nse', 'ssl-cert-intaddr.nse',
        'ssl-date.nse', 'ssl-dh-params.nse', 'ssl-enum-ciphers.nse',
        'ssl-heartbleed.nse', 'ssl-known-key.nse', 'ssl-poodle.nse',
        'sslv2.nse', 'sslv2-drown.nse', 'tls-alpn.nse',
        'tls-nextprotoneg.nse', 'tls-ticketbleed.nse'
    ],
    'vnc': [
        'vnc-brute.nse', 'vnc-info.nse', 'vnc-title.nse',
        'realvnc-auth-bypass.nse'
    ],
    'telnet': [
        'telnet-brute.nse', 'telnet-encryption.nse', 'telnet-ntlm-info.nse'
    ],
    'pop3': [
        'pop3-brute.nse', 'pop3-capabilities.nse', 'pop3-ntlm-info.nse'
    ],
    'imap': [
        'imap-brute.nse', 'imap-capabilities.nse', 'imap-ntlm-info.nse'
    ],
    'sip': [
        'sip-brute.nse', 'sip-call-spoof.nse', 'sip-enum-users.nse',
        'sip-methods.nse'
    ],
    'ajp': [
        'ajp-auth.nse', 'ajp-brute.nse', 'ajp-headers.nse',
        'ajp-methods.nse', 'ajp-request.nse'
    ],
    'afp': [
        'afp-brute.nse', 'afp-ls.nse', 'afp-path-vuln.nse',
        'afp-serverinfo.nse', 'afp-showmount.nse'
    ],
    'oracle': [
        'oracle-brute.nse', 'oracle-brute-stealth.nse', 'oracle-enum-users.nse',
        'oracle-sid-brute.nse', 'oracle-tns-version.nse'
    ],
    'mongodb': [
        'mongodb-brute.nse', 'mongodb-databases.nse', 'mongodb-info.nse'
    ],
    'redis': [
        'redis-brute.nse', 'redis-info.nse'
    ],
    'cassandra': [
        'cassandra-brute.nse', 'cassandra-info.nse'
    ],
    'docker': [
        'docker-version.nse'
    ],
    'wordpress': [
        'http-wordpress-brute.nse', 'http-wordpress-enum.nse',
        'http-wordpress-users.nse'
    ],
    'joomla': [
        'http-joomla-brute.nse'
    ],
    'drupal': [
        'http-drupal-enum.nse', 'http-drupal-enum-users.nse'
    ],
    'ntp': [
        'ntp-info.nse', 'ntp-monlist.nse'
    ],
    'ike': [
        'ike-version.nse'
    ],
    'tftp': [
        'tftp-enum.nse', 'tftp-version.nse'
    ],
    'irc': [
        'irc-botnet-channels.nse', 'irc-brute.nse', 'irc-info.nse',
        'irc-sasl-brute.nse', 'irc-unrealircd-backdoor.nse'
    ],
    'ldap': [
        'ldap-brute.nse', 'ldap-novell-getpass.nse', 'ldap-rootdse.nse',
        'ldap-search.nse'
    ],
    'rpc': [
        'rpc-grind.nse', 'rpcinfo.nse'
    ],
    'rsync': [
        'rsync-brute.nse', 'rsync-list-modules.nse'
    ],
    'svn': [
        'svn-brute.nse', 'http-svn-enum.nse', 'http-svn-info.nse'
    ],
    'vmware': [
        'http-vmware-path-vuln.nse', 'vmware-version.nse'
    ],
    'weblogic': [
        'weblogic-t3-info.nse'
    ],
    'upnp': [
        'upnp-info.nse', 'broadcast-upnp-info.nse'
    ],
    'dhcp': [
        'dhcp-discover.nse', 'broadcast-dhcp-discover.nse',
        'broadcast-dhcp6-discover.nse'
    ],
    'nbns': [
        'nbns-interfaces.nse', 'nbstat.nse'
    ],
    'bacnet': [
        'bacnet-info.nse'
    ],
    'modbus': [
        'modbus-discover.nse'
    ],
    's7': [
        's7-info.nse'
    ],
    'profinet': [
        'profinet-cm-lookup.nse', 'multicast-profinet-discovery.nse'
    ]
}


def command(target):
    if not target or "Open Ports" not in target or not target["Open Ports"]:
        return "nmap -sV <target>"  # Fallback if no port data

    open_ports = target["Open Ports"]
    selected_scripts = set()  # Avoid duplicates

    # Step 1: Check each service in service_to_scripts against open ports
    for service_name, scripts in service_to_scripts.items():
        for port, service_description in open_ports.items():
            if service_name.lower() in service_description.lower():
                selected_scripts.update(scripts)  # Add all scripts for this service

    # Step 2: If no scripts matched, fall back to a basic scan
    if not selected_scripts:
        return f"nmap -sV -p {','.join(open_ports.keys())} {target['Target']}"

    # Step 3: Build the Nmap command
    nmap_command = f"nmap --script {','.join(selected_scripts)} {target['Target']}"

    return nmap_command
test = {'Target': 'localhost', 'Iden': 'URL', 'Stat': 'up', 'Open Ports': {'135': 'Microsoft Windows RPC', '2869': 'Microsoft HTTPAPI httpd 2.0 (SSDP/UPnP)', '5357': 'Microsoft HTTPAPI httpd 2.0 (SSDP/UPnP)', '9010': 'WebSocket++ 0.8.2'}, 'OS': 'windows'}
print(command(test))
