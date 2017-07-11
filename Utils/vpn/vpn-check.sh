#!/bin/bash
ping -c3 192.168.51.140 > pingreport
result="$(ifconfig ppp0 | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}')"
if [[ $result != *'192'* ]]; then
pon labranet
sleep 10
python3 /home/pi/dev/AIScouts/RaspberryVersion/send_ppp_address.py
fi
result="$(pgrep -fl rasp_run_detection.py)"
if [[ $result != *'python3'* ]]; then
python3 /home/pi/dev/AIScouts/RaspberryVersion/rasp_run_detection.py
fi
