#!/usr/bin/env python
import requests
import subprocess
import json

ip = subprocess.getoutput("/sbin/ifconfig ppp0 | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}'")
r = requests.post('http://192.168.51.140:8080/api/v1/Eo8KxecNVvn9AVg3VXjS/telemetry',
                                  data=json.dumps({'pppAdress': ip}))


