import pytest

from pymlkit import Activations, Losses, Network


def test_forward_returns_output_sized_list():
    net = Network(2)
    net.add_dense(4, Activations.ReLU)
    net.add_dense(3, Activations.Linear)
    out = net.forward([0.5, -0.5])
    assert len(out) == 3


def test_wrong_input_size_raises():
    net = Network(2)
    net.add_dense(1)
    with pytest.raises(ValueError):
        net.forward([1.0])


def test_learns_and_gate():
    net = Network(2)
    net.add_dense(4, Activations.ReLU)
    net.add_dense(1, Activations.Sigmoid)
    net.set_loss(Losses.HalfMSE)
    x = [0, 0, 0, 1, 1, 0, 1, 1]
    y = [0, 0, 0, 1]
    net.fit(x, y, lr=0.5, epochs=2000, batch=1, log_step=100000)
    assert net.forward([1, 1])[0] > 0.5
    assert net.forward([0, 1])[0] < 0.5


def test_load_missing_file_returns_false(tmp_path):
    net = Network(1)
    net.add_dense(1)
    assert net.load(str(tmp_path / "nope.mlk")) is False
