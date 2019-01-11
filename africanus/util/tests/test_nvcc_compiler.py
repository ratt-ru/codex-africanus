import pytest


def test_nvcc_compiler(tmpdir):
    from pkg_resources import resource_filename
    from distutils import errors
    from os.path import join as pjoin

    from africanus.util.code import format_code
    from africanus.util.nvcc import get_compiler
    from africanus.util.nvcc import get_compiler_setting
    from africanus.util.nvcc import get_cuda_devices
    from africanus.util.nvcc import _get_compiler_base_options
    from africanus.util.nvcc import check_cuda_version, check_cudnn_version
    from africanus.util.nvcc import get_cuda_version, get_cudnn_version

    cc = get_compiler()(None)
    settings = get_compiler_setting()

    if check_cuda_version(cc, settings):
        print("CUDA VERSION", get_cuda_version())
    else:
        pytest.skip("nvcc not available")

    include_path = resource_filename("africanus", "include")
    cub_path = pjoin(include_path, "cub")

    code = """
    #include <cub/cub.cuh>

    __global__ void kernel(int * in)
    {

    }

    """
    tmpdir = str(tmpdir)

    tmpfile = pjoin(tmpdir, "test.cu")

    with open(tmpfile, "w") as f:
        f.write(code)

    include_dirs = settings['include_dirs'] = [cub_path]

    try:
        objects = cc.compile([tmpfile], output_dir=tmpdir,
                             include_dirs=include_dirs,
                             macros=None)
    except errors.CompileError as e:
        print(format_code(code))
        print(e)
    else:
        print(objects)
