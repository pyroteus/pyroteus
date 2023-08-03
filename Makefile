all: install

.PHONY: demos test

install:
	@echo "Installing dependencies..."
	@python3 -m pip install -r requirements.txt
	@echo "Done."
	@echo "Installing Goalie..."
	@python3 -m pip install -e .
	@echo "Done."

lint:
	@echo "Checking lint..."
	@flake8 --ignore=E501,E226,E402,E741,F403,F405,W503
	@echo "PASS"

test: lint
	@echo "Running test suite..."
	@cd test && make
	@cd test_adjoint && make
	@echo "PASS"

coverage:
	@echo "Generating coverage report..."
	@python3 -m coverage erase
	@python3 -m coverage run -a --source=goalie -m pytest -v test
	@python3 -m coverage run -a --source=goalie -m pytest -v test_adjoint
	@python3 -m coverage html

demo:
	@echo "Running all demos..."
	@cd demos && make
	@echo "Done."

doc: demo
	@echo "Building docs in html format..."
	@cd docs && make html
	@echo "Done."

tree:
	@tree -d .

clean:
	@cd demos && make clean
	@cd test && make clean
