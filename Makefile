.PHONY: clean	clean-build clean-pyc docs help

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr .cache/
	rm -fr .ipynb_checkpoints/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

test:
	pytest --flake8

docs:
	rm -f docs/proton_decay_study.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ proton_decay_study
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

install: clean
	pip install -U pip
	pip install -U setuptools
	pip install tensorflow-gpu
	pip install -r requirements.txt
	python setup.py install

uninstall:
	pip uninstall proton_decay_study

docserver:
	cd docs/_build/html && python -m SimpleHTTPServer 9000