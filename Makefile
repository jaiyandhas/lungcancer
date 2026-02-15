.PHONY: install train test run clean docker-build

install:
	venv/bin/pip install -r requirements.txt

train:
	venv/bin/python -m src.train

test:
	venv/bin/python -m pytest tests/

run:
	venv/bin/streamlit run app/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete

docker-build:
	docker build -t lung-cancer-app .
