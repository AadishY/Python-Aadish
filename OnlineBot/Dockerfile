# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY OnlineBot/requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY OnlineBot/ .

# Expose the port Streamlit is running on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"]
