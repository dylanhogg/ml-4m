env:
	# https://github.com/dylanhogg/ml-4m/blob/main/README.md#installation
	conda create -n fourm python=3.9 -y
	conda activate fourm
	# which python: /Users/dylan/miniconda3/envs/fourm/bin/python
	pip install --upgrade pip  # enable PEP 660 support (Successfully installed pip-24.1.1)
	pip install -e .

dh-example:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python dh_example.py
	# python dh_example.py
