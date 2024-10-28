# coding=utf-8
# Gunicorn configuration variables
bind = "0.0.0.0:7000"
workers = 3
timeout = 90
accesslog = "storage/access.log"  # Access logs file
errorlog = "-"    # Disable gunicorn access logs
loglevel = "info"
