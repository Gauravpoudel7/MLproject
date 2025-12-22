FROM python:3.11

WORKDIR /detecting_fraud_web

COPY . /detecting_fraud_web

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit","run","detecting_fraud_web.py"]