from __future__ import print_function

import cupy
from cupy import RawKernel
from jinja2 import Template

PHASE_DELAY_KERNEL = Template(r"""
#include "cuda.h"

using Real = {{RealType}};
using Complex = {{RealType}}2;

#define BLOCKDIMX {{BLOCKDIMX}}
#define BLOCKDIMY {{BLOCKDIMY}}

__global__ phase_delay(
    const Real * lm,
    const Real * uvw,
    const Real * frequency,
    Complex * out)
{
}
""")


code = PHASE_DELAY_KERNEL.render(RealType='float', BLOCKDIMX=16, BLOCKDIMY=16)
code = code.encode("utf-8")

print(code)

kernel = RawKernel(code.encode("utf-8"), "phase_delay")
x1 = cupy.arange(100, dtype=cupy.float32).reshape(10, 10)
x2 = cupy.ones((10, 10), dtype=cupy.float32)
y = cupy.zeros((10, 10), dtype=cupy.float32)
# kernel((10,), (10,), (x1, x2, y))
print(kernel)


