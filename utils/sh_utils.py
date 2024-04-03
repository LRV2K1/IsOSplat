#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import isosplat.cuda as _C

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

C0 = 0.28209479177387814


def num_sh_bases(degree: int):
    if degree == 0:
        return 1
    if degree == 1:
        return 4
    if degree == 2:
        return 9
    if degree == 3:
        return 16
    return 25


def deg_from_sh(num_bases: int):
    if num_bases == 1:
        return 0
    if num_bases == 4:
        return 1
    if num_bases == 9:
        return 2
    if num_bases == 16:
        return 3
    if num_bases == 25:
        return 4
    assert False, "Invalid number of SH bases"


def spherical_harmonics(
    degrees_to_use: int,
    viewdirs: Float[Tensor, "*batch 3"],
    coeffs: Float[Tensor, "*batch D C"],
) -> Float[Tensor, "*batch C"]:
    """Compute spherical harmonics

    Note:
        This function is only differentiable to the input coeffs.

    Args:
        degrees_to_use (int): degree of SHs to use (<= total number available).
        viewdirs (Tensor): viewing directions.
        coeffs (Tensor): harmonic coefficients.

    Returns:
        The spherical harmonics.
    """
    assert coeffs.shape[-2] >= num_sh_bases(degrees_to_use)
    return _SphericalHarmonics.apply(
        degrees_to_use, viewdirs.contiguous(), coeffs.contiguous()
    )


class _SphericalHarmonics(Function):
    """Compute spherical harmonics

    Args:
        degrees_to_use (int): degree of SHs to use (<= total number available).
        viewdirs (Tensor): viewing directions.
        coeffs (Tensor): harmonic coefficients.
    """

    @staticmethod
    def forward(
        ctx,
        degrees_to_use: int,
        viewdirs: Float[Tensor, "*batch 3"],
        coeffs: Float[Tensor, "*batch D C"],
    ):
        num_points = coeffs.shape[0]
        ctx.degrees_to_use = degrees_to_use
        degree = deg_from_sh(coeffs.shape[-2])
        ctx.degree = degree
        ctx.save_for_backward(viewdirs)
        return _C.compute_sh_forward(
            num_points, degree, degrees_to_use, viewdirs, coeffs
        )

    @staticmethod
    def backward(ctx, v_colors: Float[Tensor, "*batch 3"]):
        degrees_to_use = ctx.degrees_to_use
        degree = ctx.degree
        viewdirs = ctx.saved_tensors[0]
        num_points = v_colors.shape[0]
        return (
            None,
            None,
            _C.compute_sh_backward(
                num_points, degree, degrees_to_use, viewdirs, v_colors
            ),
        )

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5