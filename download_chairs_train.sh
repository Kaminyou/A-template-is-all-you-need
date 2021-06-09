#!/bin/bash

# download 03001627_train.tar.gz from drive
wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=\
$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=17j9uOb3cVXm4sqHcRcgkPBFdCmsYAv3J' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17j9uOb3cVXm4sqHcRcgkPBFdCmsYAv3J" \
-O 03001627_train.tar.gz && rm -rf /tmp/cookies.txt

# extract 03001627_train.tar.gz
tar zxvf 03001627_train.tar.gz

# merge all the data to the main folder
mv 03001627/* $1/SdfSamples/ShapeNetV2/03001627/

# remove redundant files
rm  03001627_train.tar.gz
rm -r 03001627