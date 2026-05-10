import numpy
import torch

from nlp_quantization.cli import set_seeds


def test_set_seeds_is_deterministic():
    set_seeds(1234)
    t1 = torch.randn(3)
    n1 = numpy.random.rand(3)

    set_seeds(1234)
    t2 = torch.randn(3)
    n2 = numpy.random.rand(3)

    assert torch.equal(t1, t2)
    assert numpy.allclose(n1, n2)
