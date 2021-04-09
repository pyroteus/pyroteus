all: lint test

.PHONY: test

dir:
	@echo "Creating directories..."
	@mkdir -p test/plots
	@mkdir -p test/outputs/burgers
	@mkdir -p test/outputs/rossby_wave
	@mkdir -p test/outputs/solid_body_rotation

lint:
	@echo "Checking lint..."
	@flake8 --ignore=E501,E226,E402,E731,E741,F403,F405,F999,N803,N806,W503
	@echo "PASS"

test: lint
	@echo "Running test suite..."
	@pytest -v test
	@echo "PASS"

tree:
	@tree .

clean:
	@echo "Removing compiled code..."
	@rm -rf __pycache__
	@rm -rf test/__pycache__
