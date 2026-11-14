PYTHON ?= python
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: help install-mac install-linux install-transliteration test models api agent gradio key benchmark docker-build docker-up

help:
	@echo "install-mac   Create .venv and install the Apple Silicon runtime"
	@echo "install-linux Create .venv and install an NVIDIA CUDA runtime"
	@echo "install-transliteration Install optional IndicXlit Romanized-Telugu support"
	@echo "test          Run the unit test suite"
	@echo "models        Download/cache IndicF5 + Vocos + IndicXlit"
	@echo "api           Run the TTS API on 127.0.0.1:8000"
	@echo "agent         Run the voice agent on 127.0.0.1:8001"
	@echo "gradio        Run the legacy Gradio studio"
	@echo "key           Create an API key (NAME=... QUOTA=... RATE=...)"
	@echo "benchmark     Run eval/benchmark.py (requires TTS_API_KEY/TTS_VOICE_ID)"
	@echo "docker-build  Build the container image"
	@echo "docker-up     Start API + agent with Docker Compose"

install-mac:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install torch==2.11.0 torchaudio==2.11.0
	$(VENV_PYTHON) -m pip install -r requirements.txt
	$(MAKE) install-transliteration VENV=$(VENV) PYTHON=$(PYTHON)

install-linux:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchaudio==2.11.0
	$(VENV_PYTHON) -m pip install -r requirements.txt
	$(MAKE) install-transliteration VENV=$(VENV) PYTHON=$(PYTHON)

install-transliteration:
	$(VENV_PYTHON) -m pip install --no-deps -r requirements-transliteration.txt
	$(VENV_PYTHON) -m pip install pydload pandas

test:
	$(VENV_PYTHON) -m unittest discover -s tests -v

models:
	$(VENV_PYTHON) -m tts.setup_models

api:
	$(VENV_PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8000

agent:
	$(VENV_PYTHON) -m uvicorn agent.server:create_app_from_env --factory --host 127.0.0.1 --port 8001

gradio:
	$(VENV_PYTHON) app.py

key:
	$(VENV_PYTHON) -m api.admin create-key --name "$(or $(NAME),development)" --quota $(or $(QUOTA),100000) --rate $(or $(RATE),120)

benchmark:
	$(VENV_PYTHON) eval/benchmark.py --voice-id "$(TTS_VOICE_ID)"

docker-build:
	docker build -t local-voice-studio:latest .

docker-up:
	docker compose up -d --build
