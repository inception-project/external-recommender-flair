# Use an official Python runtime as a parent image
FROM python:3.6

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY app_flair.py /app
COPY gunicorn.conf /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip3 install  -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
ENTRYPOINT ["gunicorn","app_flair:app","-c","gunicorn.conf"]
