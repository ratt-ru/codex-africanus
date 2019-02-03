- [ ] Tests added / passed

  ```bash
  $ py.test -v -s africanus
  ```

  If the pep8 tests fail, the quickest way to correct
  this is to run `autopep8` and then `flake8` and
  `pycodestyle` to fix the remaining issues.

  ```
  $ pip install -U autopep8 flake8 pycodestyle
  $ autopep8 -r -i africanus
  $ flake8 africanus
  $ pycodestyle africanus
  ```

- [ ] Fully documented, including `HISTORY.rst` for all changes
      and one of the `docs/*-api.rst` files for new API

  To build the docs locally:

  ```
  pip install -r requirements.readthedocs.txt
  cd docs
  READTHEDOCS=True make html
  ```
