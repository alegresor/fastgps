exportenv:
	conda env export --no-builds | tail -r | tail -n +2 | tail -r > conda_env.yml