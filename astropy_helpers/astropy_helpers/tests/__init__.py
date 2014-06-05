import os
import subprocess as sp
import sys

from setuptools.sandbox import run_setup

import pytest

PACKAGE_DIR = os.path.dirname(__file__)


def run_cmd(cmd, args, path=None, raise_error=True):
    """
    Runs a shell command with the given argument list.  Changes directory to
    ``path`` if given, otherwise runs the command in the current directory.

    Returns a 3-tuple of (stdout, stderr, exit code)

    If ``raise_error=True`` raise an exception on non-zero exit codes.
    """

    if path is not None:
        # Transparently support py.path objects
        path = str(path)

    p = sp.Popen([cmd] + list(args), stdout=sp.PIPE, stderr=sp.PIPE,
                 cwd=path)
    streams = tuple(s.decode('latin1').strip() for s in p.communicate())
    return_code = p.returncode

    if raise_error and return_code != 0:
        raise RuntimeError(
            "The command `{0}` with args {1!r} exited with code {2}.\n"
            "Stdout:\n\n{3}\n\nStderr:\n\n{4}".format(
                cmd, list(args), return_code, streams[0], streams[1]))

    return streams + (return_code,)


@pytest.fixture(scope='function', autouse=True)
def reset_setup_helpers(request):
    """
    Saves and restores the global state of the astropy_helpers.setup_helpers
    module between tests.
    """

    mod = __import__('astropy_helpers.setup_helpers', fromlist=[''])

    old_state = mod._module_state.copy()

    def finalizer(old_state=old_state):
        mod = sys.modules.get('astropy_helpers.setup_helpers')
        if mod is not None:
            mod._module_state.update(old_state)

    request.addfinalizer(finalizer)


@pytest.fixture(scope='function', autouse=True)
def reset_distutils_log():
    """
    This is a setup/teardown fixture that ensures the log-level of the
    distutils log is always set to a default of WARN, since different
    settings could affect tests that check the contents of stdout.
    """

    from distutils import log
    log.set_threshold(log.WARN)


@pytest.fixture
def package_template(tmpdir, request):
    """Create a copy of the package_template repository (containing the package
    template) in a tempdir and change directories to that temporary copy.

    Also ensures that any previous imports of the test package are unloaded
    from `sys.modules`.
    """

    tmp_package = tmpdir.join('package_template')

    # TODO: update URL once package-template changes are merged
    run_cmd('git', ['clone', 'http://github.com/astropy/package-template',
                    str(tmp_package)])

    old_cwd = os.getcwd()

    # Before changing directores import the local ah_boostrap module so that it
    # is tested, and *not* the copy that happens to be included in the test
    # package

    import ah_bootstrap

    # This is required to prevent the multiprocessing atexit bug
    import multiprocessing

    os.chdir(str(tmp_package))

    if 'packagename' in sys.modules:
        del sys.modules['packagename']

    old_astropy_helpers = None
    if 'astropy_helpers' in sys.modules:
        # Delete the astropy_helpers that was imported by running the tests so
        # as to not confuse the astropy_helpers that will be used in testing
        # the package
        old_astropy_helpers = sys.modules['astropy_helpers']
        del sys.modules['astropy_helpers']

    if '' in sys.path:
        sys.path.remove('')

    sys.path.insert(0, '')

    def finalize(old_cwd=old_cwd, old_astropy_helpers=old_astropy_helpers):
        os.chdir(old_cwd)
        sys.modules['astropy_helpers'] = old_astropy_helpers

    request.addfinalizer(finalize)

    return tmp_package


TEST_PACKAGE_SETUP_PY = """\
#!/usr/bin/env python

from setuptools import setup

NAME = 'astropy-helpers-test'
VERSION = {version!r}

setup(name=NAME, version=VERSION,
      packages=['_astropy_helpers_test_'],
      zip_safe=False)
"""


@pytest.fixture
def testpackage(tmpdir, version='0.1'):
    """
    This fixture creates a simplified package called _astropy_helpers_test_
    used primarily for testing ah_boostrap, but without using the
    astropy_helpers package directly and getting it confused with the
    astropy_helpers package already under test.
    """

    source = tmpdir.mkdir('testpkg')

    with source.as_cwd():
        source.mkdir('_astropy_helpers_test_')
        init = source.join('_astropy_helpers_test_', '__init__.py')
        init.write('__version__ = {0!r}'.format(version))
        setup_py = TEST_PACKAGE_SETUP_PY.format(version=version)
        source.join('setup.py').write(setup_py)

        # Make the new test package into a git repo
        run_cmd('git', ['init'])
        run_cmd('git', ['add', '--all'])
        run_cmd('git', ['commit', '-m', 'test package'])

    return source


# Ugly workaround
# Note sure exactly why, but there is some weird interaction between setuptools
# entry points and the way run_setup messes with sys.modules that causes this
# module go out out of scope during the tests; importing it here prevents that
import setuptools.py31compat
