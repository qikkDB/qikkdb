#!/bin/bash
SCRIPTPATH=$(dirname $(realpath $0) )
cd "$SCRIPTPATH"/../ColmnarDB.ConsoleClient
dotnet publish -c Release -r linux-x64 -o ../publish/console