- [ ] Tests added / passed

  ```bash
  $ py.test -v -s africanus
  ```

  If the pre-commit tests fail, install and
  run the pre-commit hooks in your development
  virtuale environment:

  ```
  $ pip install pre-commit
  $ pre-commit install
  $ pre-commit run -a
  ```

- [ ] Fully documented, including `HISTORY.rst` for all changes
      and one of the `docs/*-api.rst` files for new API

  To build the docs locally:

  ```
  pip install -r requirements.readthedocs.txt
  cd docs
  READTHEDOCS=True make html
  ```
