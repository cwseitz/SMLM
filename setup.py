from distutils.core import setup, Extension
import numpy

def main():

    setup(name="SMLM",
          version="1.0.0",
          description="Single Molecule Localization Microscopy",
          author="Clayton Seitz",
          author_email="cwseitz@iu.edu",
          ext_modules=[Extension("SMLM._ssa", ["SMLM/_ssa/ssa.c"],
                       include_dirs = [numpy.get_include(), '/usr/include/gsl'],
                       library_dirs = ['/usr/lib/x86_64-linux-gnu'],
                       libraries=['m', 'gsl', 'gslcblas'])])


if __name__ == "__main__":
    main()
