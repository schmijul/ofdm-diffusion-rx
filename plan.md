# OFDM Diffusion Denoiser — Project Plan

## Goal
Build a PyTorch-based OFDM simulation that compares classical receiver performance (LS/MMSE equalization) against a DDPM-based post-equalization denoiser. The diffusion model takes MMSE-equalized symbols and iteratively denoises them toward clean constellation points.

Target: Reproduce the core idea from the CDDM paper (Wu et al., IEEE TWC 2024) in a clean, modular codebase.

## Scope
- **Modulation:** 16-QAM
- **Waveform:** OFDM (CP-OFDM)
- **Channel model:** Rayleigh multipath fading (start with 3GPP TDL-A, fallback to simple exponential PDP)
- **Baselines:** LS + ZF, LS + MMSE, MMSE with perfect CSI (genie bound)
- **Diffusion model:** DDPM-based denoiser operating on MMSE-equalized symbols in frequency domain
- **Primary metric:** BER vs SNR curves
- **Secondary metrics:** Constellation plots (before/after denoising), SER, training convergence

## Repo Structure

```
ofdm-diffusion-denoiser/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml          # All hyperparameters (OFDM, channel, training, diffusion)
├── src/
│   ├── __init__.py
│   ├── ofdm.py                # OFDM modulator/demodulator (IFFT, CP, FFT, pilot insertion)
│   ├── channel.py             # Channel models (Rayleigh multipath, AWGN, TDL-A)
│   ├── estimation.py          # Channel estimation (LS, MMSE)
│   ├── equalization.py        # Equalization (ZF, MMSE)
│   ├── demapper.py            # Hard/soft QAM demapping
│   ├── classical_receiver.py  # End-to-end classical pipeline combining the above
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── noise_schedule.py  # Beta schedule, alpha_bar computation
│   │   ├── model.py           # U-Net style denoiser (adapted for 1D complex signal)
│   │   └── ddpm.py            # Forward/reverse diffusion process, sampling
│   ├── dataset.py             # Generate training pairs: (noisy equalized symbols, clean symbols)
│   └── metrics.py             # BER, SER computation
├── scripts/
│   ├── train.py               # Train the diffusion denoiser
│   ├── evaluate.py            # Run BER vs SNR comparison across all methods
│   └── plot_results.py        # Generate publication-quality plots
└── results/                   # Output directory for plots and metrics
```

## OFDM System Parameters (default.yaml)

```yaml
ofdm:
  n_subcarriers: 64
  cp_length: 16
  n_pilot_subcarriers: 8        # Evenly spaced pilots
  pilot_pattern: "comb"         # comb or block
  n_ofdm_symbols: 14            # Per frame (1 slot)

modulation:
  order: 16                     # 16-QAM
  
channel:
  model: "rayleigh"             # "rayleigh" | "tdl_a"
  n_taps: 8
  max_delay_spread: 5e-6        # seconds
  doppler_hz: 0                 # Start static, add mobility later
  
snr_range_db: [-5, 30]          # For evaluation sweep
snr_step_db: 2.5

training:
  snr_train_range_db: [0, 25]   # SNR range seen during training
  n_train_samples: 100000       # (equalized_symbol, clean_symbol) pairs
  n_val_samples: 10000
  batch_size: 256
  epochs: 100
  lr: 1e-4
  optimizer: "adam"

diffusion:
  n_timesteps: 200              # Forward process steps
  beta_start: 1e-4
  beta_end: 0.02
  schedule: "linear"            # "linear" | "cosine"
  inference_steps: 50           # Can be < n_timesteps with DDIM
  
  model:
    type: "unet_1d"
    channels: [64, 128, 256]
    time_embedding_dim: 128
    attention_layers: [2]       # Apply attention at resolution level 2
    residual_blocks_per_level: 2
```

## Implementation Phases

### Phase 1: Classical OFDM Baseline
Build the full classical pipeline and validate against theory.

Files: `ofdm.py`, `channel.py`, `estimation.py`, `equalization.py`, `demapper.py`, `classical_receiver.py`, `metrics.py`

Validation checks:
- [ ] OFDM modulation/demodulation with AWGN-only channel gives 0 BER at high SNR
- [ ] LS estimation error matches theoretical MSE = σ² / |X_pilot|²
- [ ] BER curves for LS+ZF and LS+MMSE match known 16-QAM curves in Rayleigh fading
- [ ] Perfect CSI + MMSE gives the genie lower bound

**Do NOT proceed to Phase 2 until all checks pass.**

### Phase 2: Dataset Generation
Generate training data by running the classical pipeline and capturing intermediate signals.

File: `dataset.py`

For each training sample:
1. Generate random 16-QAM symbols
2. OFDM modulate → channel → OFDM demodulate
3. LS channel estimation → MMSE equalization
4. Store pair: (MMSE_equalized_symbol [complex], clean_transmitted_symbol [complex])

Represent complex values as 2-channel real tensors: [Re, Im] shape (2,).

Important: Generate data across a range of SNR values (0–25 dB uniform). The model must see different noise levels during training.

### Phase 3: Diffusion Model
Implement the DDPM denoiser operating on equalized symbols.

Files: `diffusion/noise_schedule.py`, `diffusion/model.py`, `diffusion/ddpm.py`

Architecture notes:
- Input: 2D real vector (Re, Im) of one equalized subcarrier symbol + timestep embedding + SNR embedding
- The model predicts the noise ε added at timestep t
- Process symbols per-subcarrier (each subcarrier is independent after OFDM demod with sufficient CP)
- Can batch all subcarriers across all OFDM symbols together — they're independent flat-fading channels
- U-Net is overkill for a scalar 2D input. Use a **residual MLP** with time/SNR conditioning:
  - 4-6 residual blocks, hidden dim 256
  - Sinusoidal time embedding (dim 128)
  - SNR conditioning via FiLM (feature-wise linear modulation)
  - Input: [Re, Im] → Output: [ε_Re, ε_Im]

Key design choice from CDDM paper: At inference, do NOT start from pure Gaussian noise. Start from the MMSE-equalized signal and run a **partial** reverse process. The equalized signal already has most of the channel effect removed — you only need to denoise the residual. This means:
- Estimate the effective noise level of the equalized signal from the SNR
- Map that noise level to a diffusion timestep t_start < T
- Run reverse process from t_start to 0

### Phase 4: Training
File: `scripts/train.py`

- Standard DDPM training: sample (x_clean, x_equalized), sample random t, add noise to x_clean at level t, predict noise
- Loss: MSE on predicted vs actual noise
- Log training loss per epoch
- Save best model checkpoint by validation loss
- Save training curves to results/

### Phase 5: Evaluation & Comparison
File: `scripts/evaluate.py`

For each SNR in [-5, 30] dB with step 2.5 dB:
1. Generate N=10000 test OFDM frames
2. Run through classical pipeline → get BER for: LS+ZF, LS+MMSE, PerfectCSI+MMSE
3. Run MMSE-equalized symbols through diffusion denoiser → hard demap → get BER
4. Store all results to CSV

File: `scripts/plot_results.py`

Generate:
- BER vs SNR comparison plot (all methods on one figure, semilogy)
- Constellation plot grid: TX / after channel / after MMSE / after diffusion (pick 3 SNR values: low/mid/high)
- Save as PNG and PDF to results/

## Technical Notes for the Agent

1. **Complex number handling:** PyTorch works with real tensors. Always split complex into [Re, Im] channels. Provide utility functions `complex_to_real(z) -> (2, N)` and `real_to_complex(x) -> (N,)` complex tensor.

2. **Reproducibility:** Set all random seeds (torch, numpy, python) in config. Use `torch.manual_seed()` and `torch.backends.cudnn.deterministic = True`.

3. **No external OFDM libraries.** Implement from scratch using torch.fft. This keeps dependencies minimal and makes the code self-contained.

4. **Channel model:** Start with simple Rayleigh (random complex Gaussian taps with exponential PDP). Only add 3GPP TDL models if Phase 1 baselines work correctly.

5. **Pilot structure:** Use comb-type pilots (every 8th subcarrier in frequency). Interpolate channel estimate between pilots using linear interpolation for LS, or proper MMSE interpolation.

6. **The diffusion model operates per-subcarrier.** Each subcarrier after CP-OFDM sees y[k] = H[k]·x[k] + n[k]. After equalization: x̂[k] = x[k] + residual_noise[k]. The diffusion model denoises this residual. Batch all subcarriers together — they are independent samples.

7. **SNR-conditional model is critical.** The residual noise variance after MMSE equalization depends on SNR. The model must know the SNR to calibrate its denoising strength. Use FiLM conditioning.

8. **Warm-start inference:** Map post-MMSE residual noise variance to diffusion timestep via σ²_t = (1 - ᾱ_t) / ᾱ_t. Find t_start where σ²_t ≈ estimated residual variance. Start reverse process there.

9. **Dependencies:** PyTorch, NumPy, Matplotlib, PyYAML, tqdm. Nothing else.

10. **GPU support:** Use `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` everywhere. Training should work on a single consumer GPU (RTX 3060+ level).
```

## Success Criteria
- Classical baselines match theoretical BER curves (within 0.5 dB)
- Diffusion denoiser shows measurable BER improvement over MMSE at low-to-mid SNR (0–15 dB)
- Clean, well-documented code that could become a blog post
- All plots generated automatically via `python scripts/evaluate.py && python scripts/plot_results.py`
