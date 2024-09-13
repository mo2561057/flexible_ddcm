name: vocational-housing

channels:
  - conda-forge
  - opensourceeconomics

dependencies:
  - python >=3.9

  # Conda
  - anaconda-client
  - conda-verify

  # Package dependencies
  - estimagic
  - numpy
  - scipy
  - pandas
  - statsmodels

  # Misc
  - jupyterlab
  - matplotlib
  - seaborn
  - lmfit
  - pdbpp
  - pip
  - pre-commit
  - pydot
  - pytask
  - pytask-latex
  - pytask-parallel
  - pytest
  - pytest-cov
  - scikit-learn
  - tox-conda
