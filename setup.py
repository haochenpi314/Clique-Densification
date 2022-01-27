from distutils.core import setup, Extension
from glob import glob

# setup(name='networkt', version='0.1',
#       packages=['networkt',
#                 'networkt.core',
#                 'networkt.algorithms',
#                 'networkt.algorithms.temporal',
#                 'networkt.generators'],
#       install_requires=["numpy", "pandas", "scipy",
#                         "natsort", "pybind11", "networkx"],
#       ext_modules=[
#           Extension('networkt.algorithms.temporal.clique_counter',
#                     ['networkt/algorithms/temporal/clique_counter/main.cpp'],
#                     extra_compile_args=['-fopenmp', '-std=c++17', '-Wall'],
#                     extra_link_args=['-lgomp'],
#                     # include_dirs=["include/boost"],
#                     language="c++")
#       ]
#       )

setup(name='networkt', version='0.1',
      packages=['networkt',
                'networkt.core',
                'networkt.algorithms',
                'networkt.generators'],
      install_requires=["numpy", "pandas", "scipy",
                        "natsort"]
      )
