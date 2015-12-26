from numpy.distutils.misc_util import get_numpy_include_dirs
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("example", ["example.pyx"],include_dirs=get_numpy_include_dirs(), language="c")]
setup(
  name = 'Memory leak',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
