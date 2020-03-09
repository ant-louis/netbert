FROM python:3
COPY . /app
WORKDIR /app
RUN pip install -U --proxy=http://proxy.esl.cisco.com:80/ pip
RUN pip install -r requirements.txt --proxy=http://proxy.esl.cisco.com:80/
ENTRYPOINT ["python"]
CMD ["app.py"]