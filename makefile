exportenv:
	conda env export --no-builds | tail -r | tail -n +2 | tail -r > conda_env.yml

doctests: 
	pytest --doctest-modules fastgps/ -W ignore --no-header

nbtests:
	pytest --nbval --cache-clear  docs/ --no-header --ignore docs/examples/probnum25_paper/

tests: doctests nbtests

fix_doctests:
	pytest --doctest-modules fastgps/ --accept

fix_nbtests:
	for f in $$(find docs -not -name '*-checkpoint.ipynb' -and -name '*.ipynb' -and -not -name 'probnum25_paper.ipynb'); do jupyter nbconvert --to notebook --execute --inplace "$$f"; done

fix: fix_doctests fix_nbtests

mkdocs_serve:
	cp README.md docs/index.md & mkdocs serve --livereload