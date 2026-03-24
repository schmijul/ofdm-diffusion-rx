from src.dataset import generate_symbol_dataset
from src.utils import load_config, set_seed


def test_dataset_shapes_and_snr_range():
    set_seed(4)
    cfg = load_config()
    ds = generate_symbol_dataset(cfg, n_samples=256, snr_min_db=0.0, snr_max_db=5.0)

    assert ds.x_clean.shape == (256, 2)
    assert ds.x_equalized.shape == (256, 2)
    assert ds.snr_db.shape == (256, 1)
    assert float(ds.snr_db.min()) >= 0.0
    assert float(ds.snr_db.max()) <= 5.0
