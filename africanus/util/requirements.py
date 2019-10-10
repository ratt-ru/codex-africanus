# -*- coding: utf-8 -*-


import importlib

from decorator import decorate

from africanus.util.docs import on_rtd
from africanus.util.testing import in_pytest, force_missing_pkg_exception


def _missing_packages(fn, packages, import_errors):
    if len(import_errors) > 0:
        import_err_str = "\n".join((str(e) for e in import_errors))
        return ("%s requires installation of "
                "the following packages: %s.\n"
                "%s" % (fn, packages, import_err_str))
    else:
        return ("%s requires installation of the following packages: %s. "
                % (fn, tuple(packages)))


class MissingPackageException(Exception):
    pass


def requires_optional(*requirements):
    """
    Decorator returning either the original function,
    or a dummy function raising a
    :class:`MissingPackageException` when called,
    depending on whether the supplied ``requirements``
    are present.

    If packages are missing and called within a test, the
    dummy function will call :func:`pytest.skip`.

    Used in the following way:

    .. code-block:: python

        try:
            from scipy import interpolate
        except ImportError as e:
            # https://stackoverflow.com/a/29268974/1611416, pep 3110 and 344
            scipy_import_error = e
        else:
            scipy_import_error = None

        @requires_optional('scipy', scipy_import_error)
        def function(*args, **kwargs):
            return interpolate(...)


    Parameters
    ----------
    requirements : iterable of string, None or ImportError
        Sequence of package names required by the decorated function.
        ImportError exceptions (or None, indicating their absence)
        may also be supplied and will be immediately re-raised within
        the decorator. This is useful for tracking down problems
        in user import logic.

    Returns
    -------
    callable
        Either the original function if all ``requirements``
        are available or a dummy function that throws
        a :class:`MissingPackageException` or skips a pytest.
    """
    # Return a bare wrapper if we're on readthedocs
    if on_rtd():
        def _function_decorator(fn):
            def _wrapper(*args, **kwargs):
                pass

            return decorate(fn, _wrapper)

        return _function_decorator

    have_requirements = True
    missing_requirements = []
    honour_pytest_marker = True
    actual_imports = []
    import_errors = []

    # Try imports
    for requirement in requirements:
        # Ignore
        if requirement is None:
            continue
        # Reraise any supplied ImportErrors
        elif isinstance(requirement, ImportError):
            import_errors.append(requirement)
        # An actual package, try to import it
        elif isinstance(requirement, str):
            try:
                importlib.import_module(requirement)
            except ImportError:
                missing_requirements.append(requirement)
                have_requirements = False
            else:
                actual_imports.append(requirement)
        # We should force exceptions, even if we're in a pytest test case
        elif requirement == force_missing_pkg_exception:
            honour_pytest_marker = False
        # Just wrong
        else:
            raise TypeError("requirements must be "
                            "None, strings or ImportErrors. "
                            "Received %s" % requirement)

    # Requested requirement import succeeded, but there were user
    # import errors that we now re-raise
    if have_requirements and len(import_errors) > 0:
        raise ImportError("Successfully imported %s "
                          "but the following user-supplied "
                          "ImportErrors ocurred: \n%s" %
                          (actual_imports,
                           '\n'.join((str(e) for e in import_errors))))

    def _function_decorator(fn):
        # We have requirements, return the original function
        if have_requirements:
            return fn

        # We don't have requirements, produce a failing wrapper
        def _wrapper(*args, **kwargs):
            """ Empty docstring """

            # We're running test cases
            if honour_pytest_marker and in_pytest():
                try:
                    import pytest
                except ImportError as e:
                    raise ImportError("Marked as in a pytest "
                                      "test case, but pytest cannot "
                                      "be imported! %s" % str(e))
                else:
                    msg = _missing_packages(
                        fn.__name__, missing_requirements, import_errors)
                    pytest.skip(msg)
            # Raise the exception
            else:
                msg = _missing_packages(
                    fn.__name__, missing_requirements, import_errors)
                raise MissingPackageException(msg)

        return decorate(fn, _wrapper)

    return _function_decorator
