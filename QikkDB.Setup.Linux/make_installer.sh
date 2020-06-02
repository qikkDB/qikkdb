#!/bin/bash

# SYNOPSIS
#  quoteSubst <text>
quoteSubst() {
  IFS= read -d '' -r < <(sed -e ':a' -e '$!{N;ba' -e '}' -e 's/[&/\]/\\&/g; s/\n/\\&/g' <<<"$1")
  printf %s "${REPLY%$'\n'}"
}

SCRIPTPATH=$(dirname $(realpath $0) )
cd $SCRIPTPATH/..

if [ ! -f "build/qikkDB/qikkDB" ]; then
    echo "Error: qikkDB was not built in srcroot/build."
    echo "Please build project and try again."
    exit 1
fi

if [ ! -f "publish/console/ColmnarDB.ConsoleClient" ]; then
    echo "Error: Console was not published in srcroot/publish/console."
    echo "Run build_console.sh and try again."
    exit 1
fi

mkdir -p "${SCRIPTPATH}/tmp"
rm -rf "${SCRIPTPATH}"/tmp/*
mkdir -p "${SCRIPTPATH}/out"
rm -rf "${SCRIPTPATH}"/out/*
mkdir -p "${SCRIPTPATH}/tmp/bin"
mkdir -p "${SCRIPTPATH}/tmp/databases"
mkdir -p "${SCRIPTPATH}/tmp/logs"
cp build/qikkDB/qikkDB "${SCRIPTPATH}/tmp/bin/qikkDB"
cp -r publish/console "${SCRIPTPATH}/tmp/"
strip --strip-all "${SCRIPTPATH}/tmp/bin/qikkDB"
cp -r configuration "${SCRIPTPATH}/tmp/"
chmod 750 "${SCRIPTPATH}/tmp/bin/qikkDB"
chmod 750 "${SCRIPTPATH}/tmp/bin"
chmod 750 "${SCRIPTPATH}/tmp/databases"
chmod 750 "${SCRIPTPATH}/tmp/logs"
cd $SCRIPTPATH/tmp
tar -czvf "${SCRIPTPATH}/out/qikkDB.tar.gz" *
cd $SCRIPTPATH/..
rm -rf "${SCRIPTPATH}/tmp"
cp "${SCRIPTPATH}/install.sh.conf" "${SCRIPTPATH}/out/install.sh"
BASE64_TAR=$(base64 -w0 "${SCRIPTPATH}/out/qikkDB.tar.gz")
BASE64_TAR_ESC=$(quoteSubst "$BASE64_TAR")
TERMS_OF_USE=$(cat "${SCRIPTPATH}/TERMS_OF_USE.txt")
TERMS_OF_USE_ESC=$(quoteSubst "$TERMS_OF_USE")
SERVICE_FILE_CONTENT=$(cat "${SCRIPTPATH}/qikkDB.service")
SERVICE_FILE_CONTENT_ESC=$(quoteSubst "$SERVICE_FILE_CONTENT")
sed -i -f - "${SCRIPTPATH}/out/install.sh"<< EOF
s/###BASE64_TAR###/${BASE64_TAR_ESC}/g
EOF
sed -i -f - "${SCRIPTPATH}/out/install.sh"<< EOF
s/###SERVICE_FILE_CONTENT###/${SERVICE_FILE_CONTENT_ESC}/g
EOF
sed -i -f - "${SCRIPTPATH}/out/install.sh"<< EOF
s/###TERMS_OF_USE###/${TERMS_OF_USE_ESC}/g
EOF
rm "${SCRIPTPATH}/out/qikkDB.tar.gz"