.PHONY: install dev api ui cli embed embed-dry calibrate test lint docker-build docker-run

PYTHON ?= python

install:
	$(PYTHON) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -e .

dev:
	$(PYTHON) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"

api:
	$(PYTHON) main.py

ui:
	$(PYTHON) -m streamlit run interfaces/streamlit_app.py

cli:
	$(PYTHON) -m interfaces.cli

embed:
	$(PYTHON) -m scripts.embed

embed-dry:
	$(PYTHON) -m scripts.embed --dry-run

calibrate:
	$(PYTHON) -m scripts.calibrate

test:
	$(PYTHON) -m pytest -v

lint:
	$(PYTHON) -m ruff check .

docker-build:
	docker build -t astarbot:2.1.0 .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env astarbot:2.1.0
