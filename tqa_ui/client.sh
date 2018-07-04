#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:.:..

case $1 in
    init)
        rm -rf tqa/tqa.sqlite3
        python tqa/manage.py makemigrations
        python tqa/manage.py migrate
        sqlite3 tqa/tqa.sqlite3 '.read tqa/init.sql'
        ;;
    run)
        python tqa/manage.py runserver 0.0.0.0:8126
        ;;
    *)
        echo 'Usage: client.sh [init|run]'
        ;;
esac
