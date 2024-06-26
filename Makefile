VENV=./venv
PYTHON=$(VENV)/bin/python3

venv:
	python3 -m venv $(VENV)

setup:
	pip install -r requirements.txt
	clearml-init

train:
	PYTHONPATH=. python src/train.py configs/experiments/project-example.yaml
