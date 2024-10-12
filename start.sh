#!/bin/bash

# init SQLite DB
python3 create_sqlite_db.py

gunicorn -c gunicorn_config.py rag_gpt_app:app --timeout 90
