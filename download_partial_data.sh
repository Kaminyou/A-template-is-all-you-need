#!/bin/bash
wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=\
$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1xf8V3aHtaTNdl6Gq8inBu15MpYSHNzWa' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xf8V3aHtaTNdl6Gq8inBu15MpYSHNzWa" \
-O data.tar.gz && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=\
$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1ZZ0JBGgCotW4YwBsaZpkhomlzpJDfukB' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZZ0JBGgCotW4YwBsaZpkhomlzpJDfukB" \
-O sv2_chairs_train_partial.json && rm -rf /tmp/cookies.txt