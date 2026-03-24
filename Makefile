VENV_PY := .venv/bin/python
VENV_PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
SMOKE_OUT := results/smoke_make
FULL_OUT := results/full
BENCH_OUT := results/benchmark

.PHONY: help venv install quick-test test smoke train evaluate plot benchmark clean

help:
	@echo "Targets:"
	@echo "  make venv        - create local virtual env (.venv)"
	@echo "  make install     - install dependencies into .venv"
	@echo "  make quick-test  - run fast unit test suite"
	@echo "  make test        - alias for quick-test"
	@echo "  make smoke       - run tiny end-to-end train/eval/plot"
	@echo "  make train       - train with default config"
	@echo "  make evaluate    - evaluate with default config"
	@echo "  make plot        - generate plots from default eval CSV"
	@echo "  make benchmark   - multi-seed BER benchmark summary"
	@echo "  make clean       - remove smoke/full result folders"

venv:
	python3 -m venv .venv

install:
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PIP) install --index-url https://download.pytorch.org/whl/cpu torch
	$(VENV_PIP) install -r requirements.txt

quick-test:
	$(PYTEST) -q

test: quick-test

smoke:
	$(VENV_PY) scripts/train.py --config config/smoke.yaml --outdir $(SMOKE_OUT)
	$(VENV_PY) scripts/evaluate.py --config config/smoke.yaml --checkpoint $(SMOKE_OUT)/best_model.pt --outdir $(SMOKE_OUT)
	$(VENV_PY) scripts/plot_results.py --config config/smoke.yaml --csv $(SMOKE_OUT)/ber_results.csv --outdir $(SMOKE_OUT)

train:
	$(VENV_PY) scripts/train.py --config config/default.yaml --outdir $(FULL_OUT)

evaluate:
	$(VENV_PY) scripts/evaluate.py --config config/default.yaml --checkpoint $(FULL_OUT)/best_model.pt --outdir $(FULL_OUT)

plot:
	$(VENV_PY) scripts/plot_results.py --config config/default.yaml --csv $(FULL_OUT)/ber_results.csv --outdir $(FULL_OUT)

benchmark:
	$(VENV_PY) scripts/benchmark.py --config config/default.yaml --checkpoint $(FULL_OUT)/best_model.pt --outdir $(BENCH_OUT) --n-frames 200 --seeds 11,22,33

clean:
	rm -rf $(SMOKE_OUT) $(FULL_OUT) $(BENCH_OUT)
