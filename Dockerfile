# Start with a base image
FROM python:3.7
# Copy our application code
WORKDIR /var/app
COPY . .
COPY requirements.txt .
# Fetch app specific dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
# Expose port
EXPOSE 5000
# Start the app
CMD ["python", "manage.py", "runserver", "0.0.0.0:5000"]