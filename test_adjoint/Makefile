all: run

run:
	@echo "Running all adjoint tests..."
	@python3 -m pytest -v --durations=10 .
	@echo "Done."

clean:
	@echo "Running adjoint test caches..."
	@rm -rf __pycache__
	@echo "Done."
	@echo "Removing adjoint test outputs..."
	@rm -rf *.jpg *.png
	@rm -rf outputs*
	@echo "Done."
