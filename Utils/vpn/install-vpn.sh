#!/bin/bash
apt-get install pptp-linux
cp pptp /etc/init.d/pptp
cp vpn-check.sh /root/vpn-check.sh
cp labranet /etc/ppp/peers/labranet
update-rc.d pptp defaults
