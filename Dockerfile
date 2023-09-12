FROM python:3.9-slim

#Copy the requirements file to the container

COPY requirements.txt .
COPY main.py .



#Install any necessary dependencies

RUN pip3 install  -r requirements.txt
#Expose the port the application will run

EXPOSE 8000

# Command to run the application

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]