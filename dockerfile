# Use the official Python 3.8 image from the Docker Hub
FROM python:3.8
RUN useradd -ms /bin/sh -u 1001 app
USER app

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Copy the rest of your application code into the container, except pacman-env
COPY --chown=app:app . /app

CMD ["bash"]