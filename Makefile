VENV_PY := .venv/bin/python
VENV_PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
SMOKE_OUT := results/smoke_make
FULL_OUT := results/full
BENCH_OUT := results/benchmark
TEXT_OUT := results/text_benchmark
PRIOR_GRID := 0.1,0.2,0.3,0.4,0.5

.PHONY: help venv install doctor quick-test test smoke train evaluate plot benchmark text-prior text-benchmark text-train-real text-benchmark-real regime-compare regime-study-fast regime-study-fast-p8 regime-study-fast-p8-8seed regime-study-large regime-study-conference regime-study-smoke prior-sweep prior-sweep-large prior-sweep-smoke pilot-sweep frontend-compare summarize-regime clean

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
	@echo "  make text-prior TEXTS=comma,separated,paths - estimate real-text bit priors for 16-QAM"
	@echo "  make text-benchmark TEXT=path - run .txt transmission benchmark with leak guard"
	@echo "  make text-train-real - train text-oriented checkpoint (bit_one_prob~0.462)"
	@echo "  make text-benchmark-real TEXT=path - run text benchmark with real-text config/checkpoint"
	@echo "  make regime-compare UNIFORM=csv NONIID=csv - compare diffusion gain across priors"
	@echo "  make regime-study-fast  - train/benchmark uniform vs non-IID fast configs"
	@echo "  make regime-study-fast-p8 - fast uniform vs non-IID study with 8 pilot subcarriers"
	@echo "  make regime-study-fast-p8-8seed - stronger fast_p8 confirmation with 8 seeds"
	@echo "  make regime-study-large - train/benchmark uniform vs non-IID large configs"
	@echo "  make regime-study-conference - high-evidence non-IID vs uniform run with stronger settings"
	@echo "  make regime-study-smoke - tiny end-to-end validation of the regime-study pipeline"
	@echo "  make prior-sweep        - sweep bit priors and plot diffusion gain trend (set SKIP_PLOTS=1 to speed up)"
	@echo "  make prior-sweep-large  - stronger prior sweep with larger base config and seed set"
	@echo "  make prior-sweep-smoke  - tiny validation run for prior-sweep pipeline"
	@echo "  make pilot-sweep        - diagnose LS+MMSE vs Perfect-CSI MMSE over pilot counts"
	@echo "  make frontend-compare   - compare regime-study outputs across front-end variants"
	@echo "  make summarize-regime UNIFORM=csv NONIID=csv OUTDIR=dir - summarize existing regime CSVs"
	@echo "  Add FORCE_TRAIN=1 to retrain and ignore existing checkpoints"
	@echo "  Add INF_STEPS=<n> to override diffusion inference steps in regime-study runs"
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

text-prior:
	@test -n "$(TEXTS)" || (echo "Usage: make text-prior TEXTS=path1.txt,path2.txt [MAX_BYTES=n]" && exit 1)
	$(VENV_PY) scripts/estimate_text_prior.py --texts "$(TEXTS)" $(if $(MAX_BYTES),--max-bytes-per-file $(MAX_BYTES))

text-benchmark:
	@test -n "$(TEXT)" || (echo "Usage: make text-benchmark TEXT=path/to/test.txt [TRAIN_TEXTS=comma,separated,paths]" && exit 1)
	$(VENV_PY) scripts/text_benchmark.py --text $(TEXT) --train-texts "$(TRAIN_TEXTS)" --config config/compare.yaml --checkpoint results/compare_run/best_model.pt --outdir $(TEXT_OUT) $(if $(MAX_BYTES),--max-bytes $(MAX_BYTES)) $(if $(START_BYTE),--start-byte $(START_BYTE))

text-train-real:
	$(VENV_PY) scripts/train.py --config config/compare_text_real.yaml --outdir results/compare_text_real

text-benchmark-real:
	@test -n "$(TEXT)" || (echo "Usage: make text-benchmark-real TEXT=path/to/test.txt [TRAIN_TEXTS=comma,separated,paths] [MAX_BYTES=n] [START_BYTE=n]" && exit 1)
	$(VENV_PY) scripts/text_benchmark.py --text $(TEXT) --train-texts "$(TRAIN_TEXTS)" --config config/compare_text_real.yaml --checkpoint results/compare_text_real/best_model.pt --outdir $(TEXT_OUT)_real $(if $(MAX_BYTES),--max-bytes $(MAX_BYTES)) $(if $(START_BYTE),--start-byte $(START_BYTE))

regime-compare:
	@test -n "$(UNIFORM)" || (echo "Usage: make regime-compare UNIFORM=.../benchmark_summary.csv NONIID=.../benchmark_summary.csv" && exit 1)
	@test -n "$(NONIID)" || (echo "Usage: make regime-compare UNIFORM=.../benchmark_summary.csv NONIID=.../benchmark_summary.csv" && exit 1)
	$(VENV_PY) scripts/plot_regime_comparison.py --uniform-csv "$(UNIFORM)" --non-iid-csv "$(NONIID)" --outdir results/regime_compare

regime-study-fast:
	$(VENV_PY) scripts/run_regime_study.py --uniform-config config/exp_uniform_fast.yaml --non-iid-config config/exp_non_iid_fast.yaml --outdir results/regime_study_fast --n-frames 30 --seeds 1,2 $(if $(INF_STEPS),--inference-steps $(INF_STEPS)) $(if $(SKIP_PLOTS),--skip-plots) $(if $(FORCE_TRAIN),--force-train)

regime-study-fast-p8:
	$(VENV_PY) scripts/run_regime_study.py --uniform-config config/exp_uniform_fast_p8.yaml --non-iid-config config/exp_non_iid_fast_p8.yaml --outdir results/regime_study_fast_p8 --n-frames 30 --seeds 1,2 $(if $(INF_STEPS),--inference-steps $(INF_STEPS)) $(if $(SKIP_PLOTS),--skip-plots) $(if $(FORCE_TRAIN),--force-train)

regime-study-fast-p8-8seed:
	$(VENV_PY) scripts/run_regime_study.py --uniform-config config/exp_uniform_fast_p8.yaml --non-iid-config config/exp_non_iid_fast_p8.yaml --outdir results/regime_study_fast_p8 --n-frames 40 --seeds 1,2,3,4,5,6,7,8 --inference-steps $(if $(INF_STEPS),$(INF_STEPS),20) $(if $(SKIP_PLOTS),--skip-plots) $(if $(FORCE_TRAIN),--force-train)

regime-study-large:
	$(VENV_PY) scripts/run_regime_study.py --uniform-config config/exp_uniform_large.yaml --non-iid-config config/exp_non_iid_large.yaml --outdir results/regime_study_large --n-frames 120 --seeds 1,2,3 $(if $(INF_STEPS),--inference-steps $(INF_STEPS)) $(if $(SKIP_PLOTS),--skip-plots) $(if $(FORCE_TRAIN),--force-train)

regime-study-conference:
	$(VENV_PY) scripts/run_regime_study.py --uniform-config config/exp_uniform_conference.yaml --non-iid-config config/exp_non_iid_conference.yaml --outdir results/regime_study_conference --n-frames 120 --seeds 1,2,3,4,5 $(if $(INF_STEPS),--inference-steps $(INF_STEPS)) $(if $(SKIP_PLOTS),--skip-plots) $(if $(FORCE_TRAIN),--force-train)

regime-study-smoke:
	$(VENV_PY) scripts/run_regime_study.py --uniform-config config/exp_uniform_fast.yaml --non-iid-config config/exp_non_iid_fast.yaml --outdir results/regime_study_smoke --epochs 1 --n-train 128 --n-val 64 --n-frames 2 --seeds 1 $(if $(SKIP_PLOTS),--skip-plots) $(if $(FORCE_TRAIN),--force-train)

prior-sweep:
	$(VENV_PY) scripts/prior_sweep.py --base-config config/exp_uniform_fast.yaml --priors "$(if $(PRIORS),$(PRIORS),$(PRIOR_GRID))" --outdir results/prior_sweep --n-frames 20 --seeds 1,2 $(if $(SKIP_PLOTS),--skip-plots) $(if $(FORCE_TRAIN),--force-train)

prior-sweep-large:
	$(VENV_PY) scripts/prior_sweep.py --base-config config/exp_uniform_large.yaml --priors "$(if $(PRIORS),$(PRIORS),$(PRIOR_GRID))" --outdir results/prior_sweep_large --n-frames 60 --seeds 1,2,3 $(if $(SKIP_PLOTS),--skip-plots) $(if $(FORCE_TRAIN),--force-train)

prior-sweep-smoke:
	$(VENV_PY) scripts/prior_sweep.py --base-config config/exp_uniform_fast.yaml --priors "0.2,0.5" --outdir results/prior_sweep_smoke --epochs 1 --n-train 128 --n-val 64 --n-frames 2 --seeds 1 $(if $(SKIP_PLOTS),--skip-plots) $(if $(FORCE_TRAIN),--force-train)

pilot-sweep:
	$(VENV_PY) scripts/pilot_sweep.py --config config/exp_uniform_large.yaml --pilot-counts "$(if $(PILOTS),$(PILOTS),4,8,16)" --snrs "$(if $(SNRS),$(SNRS),0,4,8,12)" --n-frames $(if $(N_FRAMES),$(N_FRAMES),60) --seeds "$(if $(SEEDS),$(SEEDS),1,2)" --outdir results/pilot_sweep

frontend-compare:
	$(VENV_PY) scripts/compare_frontend_regimes.py --regime "4 pilots:results/regime_study_fast" --regime "8 pilots:results/regime_study_fast_p8" --outdir results/frontend_compare

summarize-regime:
	@test -n "$(UNIFORM)" || (echo "Usage: make summarize-regime UNIFORM=.../benchmark_summary.csv NONIID=.../benchmark_summary.csv OUTDIR=results/..." && exit 1)
	@test -n "$(NONIID)" || (echo "Usage: make summarize-regime UNIFORM=.../benchmark_summary.csv NONIID=.../benchmark_summary.csv OUTDIR=results/..." && exit 1)
	@test -n "$(OUTDIR)" || (echo "Usage: make summarize-regime UNIFORM=.../benchmark_summary.csv NONIID=.../benchmark_summary.csv OUTDIR=results/..." && exit 1)
	$(VENV_PY) scripts/summarize_regime.py --uniform-csv "$(UNIFORM)" --non-iid-csv "$(NONIID)" --outdir "$(OUTDIR)"

clean:
	rm -rf $(SMOKE_OUT) $(FULL_OUT) $(BENCH_OUT) $(TEXT_OUT)
