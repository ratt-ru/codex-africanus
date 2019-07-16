.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/ska-sa/codex-africanus/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Codex Africanus could always use more documentation, whether as part of the
official Codex Africanus docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/ska-sa/codex-africanus/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `codex-africanus` for local development.

1. Fork the `codex-africanus` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/codex-africanus.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv codex-africanus
    $ cd codex-africanus/
    $ pip install -e .

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes
   pass the test cases, fixup your PEP8 compliance,
   and check for any code style issues:

    $ py.test -v africanus
    $ autopep8 -r -i africanus
    $ flake8 africanus
    $ pycodestyle africanus

   To get autopep8 and pycodestyle, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in HISTORY.rst.
3. The pull request should work for Python 2.7, 3.5 and 3.6. Check
   https://travis-ci.org/ska-sa/codex-africanus/pull_requests
   and make sure that the tests pass for all supported Python versions.

Tips
----

To run the tests::

$ py.test -vvv africanus/


Deploying
---------

A reminder for the maintainers on how to deploy.

1. Update HISTORY.rst with the intended release number Z.Y.X and commit to git.

2. Bump the version number with bumpversion. This creates a new git commit,
   as well as an annotated tag Z.Y.X for the release.
   If your current version is Z.Y.W and the new version is Z.Y.X call::

       $ python -m pip install bump2version
       $ bump2version --current-version Z.Y.W --new-version Z.Y.X patch

3. Push the release commit and new tag up::

       $ git push --follow-tags

4. Travis should automatically deploy the tagged release to PyPI
   if the automated tests pass.
