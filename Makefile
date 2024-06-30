env:
	# https://github.com/dylanhogg/ml-4m/blob/main/README.md#installation
	conda create -n fourm python=3.9 -y
	conda activate fourm
	# which python: /Users/dylan/miniconda3/envs/fourm/bin/python
	pip install --upgrade pip  # enable PEP 660 support (Successfully installed pip-24.1.1)
	pip install -e .

venv:
	python3 -m venv venv
	source venv/bin/activate ; pip install --upgrade pip
	# source venv/bin/activate ; pip install torch torchvision torchaudio
	source venv/bin/activate ; python3 -m pip install -r requirements.txt
	source venv/bin/activate ; pip freeze > requirements_freeze.txt

dh-example:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python dh_example.py
	# python dh_example.py

run-generation:
	# PYTORCH_ENABLE_MPS_FALLBACK=1 python run_generation.py --device cpu --model EPFL-VILAB/4M-21_B --sr_model EPFL-VILAB/4M-21_B --tokens_per_target 20 --sr_tokens_per_target 20
	PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 run_generation.py --device cpu -c cfgs/default/generation/models/4m-xl_mod7+sr_4m-l_mod7.yaml -dc cfgs/default/generation/data/parti_3x.yaml -gc cfgs/default/generation/settings_base/T2CR_roar49-25_cfg3_t6-0.5.yaml -src cfgs/default/generation/settings_sr/x2CR_mg8_cfg3_t1const.yaml


.DEFAULT_GOAL := help
help:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
