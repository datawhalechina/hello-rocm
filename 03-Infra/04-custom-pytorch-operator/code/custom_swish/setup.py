from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_custom_swish', # 安装后的包名
    ext_modules=[
        CUDAExtension(
            name='my_custom_swish_backend', # 编译生成的底层库名
            sources=['fused_swish_wrapper.cpp', 'fused_swish_kernel.hip'],
            # 开启 C++ 和 HIP 编译器的最高级别优化 -O3
            # 在 ROCm 环境下，'nvcc' 参数会被传递给 hipcc
            extra_compile_args={'cxx': ['-O3'], 'nvcc':['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
