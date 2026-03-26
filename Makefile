VENV_PY := .venv/bin/python
VENV_PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
SMOKE_OUT := results/smoke_make
FULL_OUT := results/full
BENCH_OUT := results/benchmark
TEXT_OUT := results/text_benchmark
PRIOR_GRID := 0.1,0.2,0.3,0.4,0.5

.PHONY: help venv install doctor quick-test test smoke train evaluate plot benchmark text-benchmark regime-compare regime-study-fast regime-study-large regime-study-smoke prior-sweep prior-sweep-smoke clean

help:
	@echo "Targets:"
	@echo "  make venv        - create local virtual env (.venv)"
	@echo "  make install     - install dependencies into .venv"
	@echo "  make doctor      - quick environment and dependency sanity checks"
	@echo "  make quick-test  - run fast unit test suite"
	@echo "  make test        - alias for quick-test"
	@echo "  make smoke       - run tiny end-to-end train/eval/plot"
	@echo "  make train       - train with default config"
	@echo "  make evaluate    - evaluate with default config"
	@echo "  make plot        - generate plots from default eval CSV"
	@echo "  make benchmark   - multi-seed BER benchmark summary"
	@echo "  make text-benchmark TEXT=path - run .txt transmission benchmark with leak guard"
	@echo "  make regime-compare UNIFORM=csv NONIID=csv - compare diffusion gain across priors"
	@echo "  make regime-study-fast  - train/benchmark uniform vs non-IID fast configs"
	@echo "  make regime-study-large - train/benchmark uniform vs non-IID large configs"
	@echo "  make regime-study-smoke - tiny end-to-end validation of the regime-study pipeline"
	@echo "  make prior-sweep        - sweep bit priors and plot diffusion gain trend (set SKIP_PLOTS=1 to speed up)"
	@echo "  make prior-sweep-smoke  - tiny validation run for prior-sweep pipeline"
	@echo "  make clean       - remove smoke/full result folders"

venv:
	python3 -m venv .venv

install:
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PIP) install --index-url https://download.pytorch.org/whl/cpu torch
	$(VENV_PIP) install -r requirements.txt

doctor:
	@test -x "$(VENV_PY)" || (echo "Missing .venv. Run: make venv && make install" && exit 1)
	$(VENV_PY) -c "import sys; print('python', sys.version.split()[0])"
	$(VENV_PY) -c "import torch, numpy, yaml, matplotlib, tqdm; print('deps ok')"

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

regime-study-fast:
	$(VENV_PY) scripts/run_regime_study.py --uniform-config config/exp_uniform_fast.yaml --non-iid-config config/exp_non_iid_fast.yaml --outdir results/regime_study_fast --n-frames 30 --seeds 1,2 $(if $(SKIP_PLOTS),--skip-plots)

regime-study-large:
	$(VENV_PY) scripts/run_regime_study.py --uniform-config config/exp_uniform_large.yaml --non-iid-config config/exp_non_iid_large.yaml --outdir results/regime_study_large --n-frames 120 --seeds 1,2,3 $(if $(SKIP_PLOTS),--skip-plots)

regime-study-smoke:
	$(VENV_PY) scripts/run_regime_study.py --uniform-config config/exp_uniform_fast.yaml --non-iid-config config/exp_non_iid_fast.yaml --outdir results/regime_study_smoke --epochs 1 --n-train 128 --n-val 64 --n-frames 2 --seeds 1 $(if $(SKIP_PLOTS),--skip-plots)

prior-sweep:
	$(VENV_PY) scripts/prior_sweep.py --base-config config/exp_uniform_fast.yaml --priors "$(if $(PRIORS),$(PRIORS),$(PRIOR_GRID))" --outdir results/prior_sweep --n-frames 20 --seeds 1,2 $(if $(SKIP_PLOTS),--skip-plots)

prior-sweep-smoke:
	$(VENV_PY) scripts/prior_sweep.py --base-config config/exp_uniform_fast.yaml --priors "0.2,0.5" --outdir results/prior_sweep_smoke --epochs 1 --n-train 128 --n-val 64 --n-frames 2 --seeds 1 $(if $(SKIP_PLOTS),--skip-plots)

clean:
	rm -rf $(SMOKE_OUT) $(FULL_OUT) $(BENCH_OUT) $(TEXT_OUT)
