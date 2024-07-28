experiment:
	python src/run.py
	echo done experiment
test:
	python -m pytest
	echo done test
build: experiment test
	echo done build
predict:
	python src/predict.py
	echo done predict