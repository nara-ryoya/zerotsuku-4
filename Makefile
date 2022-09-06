lint:
	export PIPENV_VERBOSITY=-1 && \
	pipenv run isort . && \
	pipenv run black . && \
	pipenv run flake8
