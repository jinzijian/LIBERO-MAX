.PHONY: test validate-pilot summarize-pilot

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

validate-pilot:
	PYTHONPATH=src python3 -m libero_max validate examples/scenarios/pilot.json

summarize-pilot:
	PYTHONPATH=src python3 -m libero_max summarize examples/results/pilot_results.jsonl --scenarios examples/scenarios/pilot.json
