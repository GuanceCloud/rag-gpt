FROM pubrepo.guance.com/chatbot/rag-gpt-base:py3.11-slim

# Set the working directory to /app inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Grant execution permissions to the start-up script
RUN chmod a+x start.sh

# Make port 7000 available to the world outside this container
EXPOSE 7000

# Define the command to run on container start. This script starts the Flask application.
CMD ["./start.sh"]
