ping -c3 192.168.51.140 > pingreport
result=`grep "0 received" pingreport`
truncresult="`echo "$result" | sed 's/^\(.................................\).*$/\1/'`"
if [[ $truncresult == "3 packets transmitted, 0 received" ]]; then
pon labranet
fi
