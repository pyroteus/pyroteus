all: run

run:
	@echo "Running all tests..."
	@python3 -m pytest -v .
	@echo "Done."

clean:
	@echo "Running test caches..."
	@rm -rf __pycache__
	@echo "Done."
	@echo "Removing test outputs..."
	@rm -rf *.jpg *.png
	@rm -rf outputs*
	@echo "Done."
