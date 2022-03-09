FROM python:3.8-slim-buster


RUN apt update -y\
  && apt install build-essential -y

COPY requirements.txt get_classla_models.py .

RUN pip3 install -r requirements.txt \
	&& python3 get_classla_models.py

COPY wiki.sl.small main.py mark_entities.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]