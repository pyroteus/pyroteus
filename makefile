all: lint test

.PHONY: test

lint:
	@echo "Checking lint..."
	@flake8 --ignore=E501,E226,E402,E731,E741,F403,F405,F999,N803,N806,W503
	@echo "PASS"

test:
	@echo "Running test suite..."
	@pytest -v test
	@echo "PASS"
