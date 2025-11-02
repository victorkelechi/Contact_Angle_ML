# Create Base Image
FROM python:3.12-slim

# Set a working directory
WORKDIR /contact_angle_app

# install pipenv
RUN pip install --no-cache-dir pipenv

# Copy Pipenv files (for caching)
COPY Pipfile Pipfile.lock /contact_angle_app/

# Install dependencies
RUN pipenv install --deploy --system

# Copy application files
COPY . /contact_angle_app/

# Expose the port the app runs on
EXPOSE 9696

# Command to run the application
CMD ["python", "predict.py"]