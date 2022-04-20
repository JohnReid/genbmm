import pytest
import torch
import genbmm


@pytest.fixture
def banded_ones():
    batch, n, off = 1, 10, 3
    return genbmm.BandedMatrix(torch.ones(batch, n, off).cuda(), lu=1, ld=1, fill=0)


def test_matmul(banded_ones):
    x = banded_ones.multiply(banded_ones)
    1 / 0

    if isinstance(a, genbmm.BandedMatrix):
        return b.multiply_log(a.transpose())
    else:
        a2, b2, size = broadcast(a, b)
    return genbmm.logbmm(a2, b2).view(size)
