#export FASTGP_DEBUG=True

exportenv:
	conda env export --no-builds | tail -r | tail -n +2 | tail -r > conda_env.yml

doctests: 
	pytest --doctest-modules fastgp/ -W ignore --no-header

nbtests:
	pytest --nbval docs/ --no-header --ignore docs/examples/probnum25_paper/

tests: doctests nbtests