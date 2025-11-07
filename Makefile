.PHONY: train evaluate serve mlflow-ui

train:
	python -m src.models.train --config config.yaml

evaluate:
	python -m src.models.evaluate --config config.yaml

serve:
	mlflow models serve -m $(MODEL_URI) -p 5001 --no-conda

mlflow-ui:
	mlflow ui --port 5000 --host 0.0.0.0
