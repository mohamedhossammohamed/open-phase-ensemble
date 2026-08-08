import numpy as np

from tsad.ingestion import StreamBuffer


def test_stream_buffer_init_and_capacity():
    buf = StreamBuffer(window_size=100)
    assert len(buf) == 0
    
    for i in range(150):
        v = buf.step(float(i))
        assert isinstance(v, float)
    
    assert len(buf) == 100
    assert buf.get_buffer()[-1] == 149.0
    assert buf.get_buffer()[0] == 50.0

def test_nan_forward_fill():
    buf = StreamBuffer(window_size=50)
    buf.step(10.0)
    buf.step(20.0)
    v_nan = buf.step(np.nan)
    
    # Should fill with last valid value 20.0
    assert not np.isnan(v_nan)
    assert buf.get_buffer()[-1] == 20.0

def test_robust_mad_normalization():
    buf = StreamBuffer(window_size=5)
    values = [10.0, 10.0, 10.0, 10.0, 10.0]
    normalized = []
    for val in values:
        v = buf.step(val)
        normalized.append(v)
        
    # Constant input should give 0.0 without division by zero NaN/Inf
    assert not np.isnan(normalized[-1])
    assert not np.isinf(normalized[-1])
    assert normalized[-1] == 0.0

def test_standard_normalization():
    buf = StreamBuffer(window_size=100)
    np.random.seed(42)
    data = np.random.normal(50.0, 10.0, size=200)
    outputs = [buf.step(x) for x in data]
    
    # Last 100 values normalized should have approx median ~ 0.0
    recent_outputs = outputs[100:]
    assert abs(np.median(recent_outputs)) < 0.5
