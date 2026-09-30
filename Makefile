.PHONY: install install-local dev api ui cli embed embed-dry calibrate check-embed test lint \
	docker-build docker-build-local docker-run

PYTHON ?= python
VERSION := 2.4.0
TORCH_CPU := --extra-index-url https://download.pytorch.org/whl/cpu

# API + UI with Hugging Face embeddings (no torch).
install:
	$(PYTHON) -m pip install -e ".[ui]"

# Adds local BGE (CPU torch) for EMBEDDING_BACKEND=local.
install-local:
	$(PYTHON) -m pip install $(TORCH_CPU) -e ".[ui,local-embed]"

# Everything: tests, lint, local BGE for `make embed` / `make calibrate`.
dev:
	$(PYTHON) -m pip install $(TORCH_CPU) -e ".[dev,ui,local-embed]"

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

# Compare HF Inference API vectors with local BGE (needs HF_TOKEN + local-embed).
check-embed:
	$(PYTHON) -m scripts.check_embedder

test:
	$(PYTHON) -m pytest -v

lint:
	$(PYTHON) -m ruff check .

# Slim image for EMBEDDING_BACKEND=hf (Railway).
docker-build:
	docker build -t astarbot:$(VERSION) .

# Fat image with torch + baked BGE for EMBEDDING_BACKEND=local.
docker-build-local:
	docker build --build-arg INSTALL_LOCAL_EMBED=true -t astarbot:$(VERSION)-local .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env astarbot:$(VERSION)
