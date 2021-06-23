from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='OrienMaskExtension',
    description='orien mask extensions',
    ext_modules=[
        CUDAExtension(
            name='eval.nms_cpu',
            sources=['eval/src/nms_cpu.cpp']
        ),
        CUDAExtension(
            name='eval.nms_cuda',
            sources=['eval/src/nms_cuda.cpp', 'eval/src/nms_kernel.cu']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
