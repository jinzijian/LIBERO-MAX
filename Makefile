.PHONY: test validate-pilot validate-physical-manifest summarize-pilot

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

validate-pilot:
	PYTHONPATH=src python3 -m libero_max validate examples/scenarios/pilot.json

validate-physical-manifest:
	PYTHONPATH=src python3 -m libero_max validate-manifest examples/manifests/cosmos_physical_pilot_v0.1.json

summarize-pilot:
	PYTHONPATH=src python3 -m libero_max summarize examples/results/pilot_results.jsonl --scenarios examples/scenarios/pilot.json
