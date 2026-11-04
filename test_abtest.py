from abtest import generate_data
import numpy as np

def test_generate_data_shapes():
    data = generate_data()
    for val in data.values():
        assert val.shape == (1000,)

def test_generate_data_reproducible():
    data1 = generate_data(seed=42)
    data2 = generate_data(seed=42)
    for key in data1.keys():
        assert np.array_equal(data1[key], data2[key])

def test_generate_data_different_seeds():
    data1 = generate_data(seed=42)
    data2 = generate_data(seed=123)
    assert not np.array_equal(data1["control_duration"], data2["control_duration"])