#!/bin/bash
SCRIPTPATH=$(dirname $(realpath $0) )
cd "$SCRIPTPATH"/../QikkDB.ConsoleClient
dotnet publish -c Release -r linux-x64 -o ../publish/console