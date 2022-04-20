"""Ad-hoc tests."""

import pytest
import torch
import genbmm


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_non_scalar_fill():
    """`BandedMatrix.multiply_log` is returning an object with a non-scalar fill
    attribute, this causes a subsequent `band_pad` to fail."""
    batch, n = 6, 3
    a_data = torch.rand((batch, n, n)).cuda()
    b_data = torch.rand((batch, n, n)).cuda()
    a = genbmm.BandedMatrix(a_data, lu=1, ld=1)
    b = genbmm.BandedMatrix(b_data, lu=1, ld=1)
    x = b.multiply_log(a.transpose())
    assert x.fill.size == 1
