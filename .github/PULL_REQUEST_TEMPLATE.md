- [ ] Tests added / passed

  ```bash
  $ py.test -v -s africanus
  ```

  If the pycodestyle tests fail, the quickest way to correct
  this is to run `autopep8` and then `pycodestyle` to fix the
  remaining issues.

  ```
  $ pip install -U autopep8 pycodestyle
  $ autopep8 -r -i africanus
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
