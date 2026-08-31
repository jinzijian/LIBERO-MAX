.PHONY: test validate-pilot validate-physical-manifest validate-lite validate-max8000 summarize-pilot

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

validate-pilot:
	PYTHONPATH=src python3 -m libero_max validate examples/scenarios/pilot.json

validate-physical-manifest:
	PYTHONPATH=src python3 -m libero_max validate-manifest examples/manifests/cosmos_physical_pilot_v0.1.json

validate-max8000:
	PYTHONPATH=src python3 -m libero_max validate-manifest benchmark/max8000/libero_max_8000.json
	cd benchmark/max8000 && shasum -a 256 -c SHA256SUMS

validate-lite:
	PYTHONPATH=src python3 -m libero_max validate-manifest benchmark/lite/libero_max_lite.json
	cd benchmark/lite && shasum -a 256 -c SHA256SUMS

summarize-pilot:
	PYTHONPATH=src python3 -m libero_max summarize examples/results/pilot_results.jsonl --scenarios examples/scenarios/pilot.json
