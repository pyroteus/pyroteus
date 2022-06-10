all: run

run:
	@echo "Running all demos..."
	@python3 time_partition.py
	@python3 mesh_seq.py
	@python3 burgers.py
	@python3 burgers1.py
	@python3 burgers2.py
	@python3 burgers_time_integrated.py
	@python3 burgers_oo.py
	@python3 burgers_ee.py
	@echo "Done."

clean:
	@echo "Running demo caches..."
	@rm -rf __pycache__
	@echo "Done."
	@echo "Removing demo outputs..."
	@rm -rf *.jpg
	@rm -rf outputs*
	@echo "Done."
