#!/bin/bash
apt-get install pptp-linux
cp pptp /etc/init.d/pptp
cp vpn-check.sh /root/vpn-check.sh
cp labranet /etc/ppp/peers/labranet
!/bin/bash

echo '"add */5  *    * * *   root  bash /root/vpn-check.sh" to crontab (/etc/crontab) to auto check for vpn'  
update-rc.d pptp defaults
