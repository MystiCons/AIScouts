#!/bin/bash
ping -c3 192.168.51.140 > pingreport
result="$(ifconfig ppp0 | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}')"
if [[ $result != *'192'* ]]; then
echo shit
pon labranet
sleep 10
python3 /home/pi/dev/AIScouts/Raspberry/send_ppp_address.py
fi
