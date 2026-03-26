VENV_PY := .venv/bin/python
VENV_PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
SMOKE_OUT := results/smoke_make
FULL_OUT := results/full
BENCH_OUT := results/benchmark
TEXT_OUT := results/text_benchmark

.PHONY: help venv install quick-test test smoke train evaluate plot benchmark text-benchmark regime-compare clean

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
	@echo "  make text-benchmark TEXT=path - run .txt transmission benchmark with leak guard"
	@echo "  make regime-compare UNIFORM=csv NONIID=csv - compare diffusion gain across priors"
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

text-benchmark:
	@test -n "$(TEXT)" || (echo "Usage: make text-benchmark TEXT=path/to/test.txt [TRAIN_TEXTS=comma,separated,paths]" && exit 1)
	$(VENV_PY) scripts/text_benchmark.py --text $(TEXT) --train-texts "$(TRAIN_TEXTS)" --config config/compare.yaml --checkpoint results/compare_run/best_model.pt --outdir $(TEXT_OUT)

regime-compare:
	@test -n "$(UNIFORM)" || (echo "Usage: make regime-compare UNIFORM=.../benchmark_summary.csv NONIID=.../benchmark_summary.csv" && exit 1)
	@test -n "$(NONIID)" || (echo "Usage: make regime-compare UNIFORM=.../benchmark_summary.csv NONIID=.../benchmark_summary.csv" && exit 1)
	$(VENV_PY) scripts/plot_regime_comparison.py --uniform-csv "$(UNIFORM)" --non-iid-csv "$(NONIID)" --outdir results/regime_compare

clean:
	rm -rf $(SMOKE_OUT) $(FULL_OUT) $(BENCH_OUT) $(TEXT_OUT)
