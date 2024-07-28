FROM python:3.10.11-slim

RUN pip install -U pip
RUN pip install poetry
RUN poetry config virtualenvs.create false
WORKDIR /app
COPY ["pyproject.toml","poetry.lock","./"]
RUN poetry install --no-dev

COPY ["src/predict.py","src/const.py","src/preprocessing.py","model/best_model/model.pickle","./"]
EXPOSE 9696
RUN mkdir src && mv predict.py const.py preprocessing.py src/ && mkdir model && mkdir model/best_model && mv model.pickle model/best_model/
ENTRYPOINT [ "gunicorn","--bind=0.0.0.0:9696","src.predict:app" ]