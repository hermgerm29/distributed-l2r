POETRY_VERSION := $(shell poetry --version 2> /dev/null)
DOCKER_VERSION := $(shell docker --version 2> /dev/null)
DOCKER_COMPOSE_VERSION := $(shell docker compose version 2> /dev/null)
NVIDIA_GPU_CHECK := $(shell nvidia-smi 2> /dev/null)

check-poetry:
ifndef POETRY_VERSION
	@echo "Please install poetry: https://python-poetry.org/docs/#installation"
	exit 1
endif

check-docker:
ifndef DOCKER_VERSION
	@echo "Please install Docker: https://docs.docker.com/get-docker/"
	exit 1
endif

check-docker-compose:
ifndef DOCKER_COMPOSE_VERSION
	@echo "Please install Docker compose: https://docs.docker.com/get-docker/"
	exit 1
endif

check-nvidia-gpu:
ifndef NVIDIA_GPU_CHECK
	@echo "Nvidia gpu not detected. Unable to run simulator image"
	exit 1
endif

.PHONY: sim-test

package: check-poetry
	@echo "\\nBuilding clean distrib-l2r package"
	rm -rf ./dist && poetry build

install: check-poetry
	@echo "\\nInstalling dependencies"
	poetry install

test: install package
	@echo "\\nRunning unit tests"
	python -m unittest discover -s test
