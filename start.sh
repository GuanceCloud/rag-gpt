#!/bin/bash

gunicorn -c gunicorn_config.py rag_gpt_app:app
