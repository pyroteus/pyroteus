all: lint test

.PHONY: demos test

install: dir
	@echo "Installing Pyroteus..."
	@python3 -m pip install -e .

lint:
	@echo "Checking lint..."
	@flake8 --ignore=E501,E226,E402,E731,E741,F403,F405,F999,N803,N806,W503
	@echo "PASS"

test: lint
	@echo "Running test suite..."
	@cd test && make
	@echo "PASS"

demo:
	@echo "Running all demos..."
	@cd demos && make
	@echo "Done."

doc: demos
	@echo "Building docs in html format..."
	@cd docs && make html
	@echo "Done."

tree:
	@tree -d .

clean:
	@cd demos && make clean
	@cd test && make clean
