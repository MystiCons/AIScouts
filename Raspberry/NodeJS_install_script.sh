#!/bin/bash

#Poistaa vanhat versiot NodeRED:st  , NodeJS:st   ja NPM:st  
#Removes old versions of the Node-RED, NodeJS and NPM
sudo apt-get remove -y nodered
sudo apt-get remove -y nodejs nodejs-legacy
sudo apt-get remove -y npm

#Lis     l  hteen ja asentaa NodeJS, olennaiset ty  kalut ja python gpio kirjaston
#Inserts source and installs NodeJS, essential tools and python GPIO lib 
sudo apt-get update
sudo apt-get install -y curl
curl -sL https://deb.nodesource.com/setup_7.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo apt-get install -y build-essential python-rpi.gpio nodejs


#Tyhjent     npm v  limuistin ja asentaa globaalisti node-REDin
#Cleans NPM cache and install node-RED globally 
sudo npm cache clean
sudo npm install -g --unsafe-perm  node-red

sudo apt-get update && sudo apt-get install python-rpi.gpio


