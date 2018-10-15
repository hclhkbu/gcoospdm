#!/bin/bash
url=$1
echo $url
wget $url
filename=${url##*/}
fn=`echo $filename | cut -d'.' -f 1`
tar xzvf $filename
