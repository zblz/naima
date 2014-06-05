import glob
import os
import textwrap

from setuptools.package_index import PackageIndex
from setuptools.sandbox import run_setup

from . import run_cmd, testpackage
from ..utils import silence


TEST_SETUP_PY = """\
#!/usr/bin/env python
from __future__ import print_function

import os
import sys

import ah_bootstrap
# reset the name of the package installed by ah_boostrap to
# _astropy_helpers_test_--this will prevent any confusion by pkg_resources with
# any already installed packages named astropy_helpers
# We also disable auto-upgrade by default
ah_bootstrap.DIST_NAME = 'astropy-helpers-test'
ah_bootstrap.PACKAGE_NAME = '_astropy_helpers_test_'
ah_bootstrap.AUTO_UPGRADE = False
try:
    ah_bootstrap.use_astropy_helpers({args})
finally:
    ah_bootstrap.DIST_NAME = 'astropy-helpers'
    ah_bootstrap.PACKAGE_NAME = 'astropy_helpers'
    ah_bootstrap.AUTO_UPGRADE = True

import _astropy_helpers_test_
filename = os.path.abspath(_astropy_helpers_test_.__file__)
filename = filename.replace('.pyc', '.py')  # More consistent this way
print(filename)
"""


def test_bootstrap_from_submodule(tmpdir, testpackage, capsys):
    """
    Tests importing _astropy_helpers_test_ from a submodule in a git
    repository.  This tests actually performing a fresh clone of the repository
    without the submodule initialized, and that importing astropy_helpers in
    that context works transparently after calling
    `ah_boostrap.use_astropy_helpers`.
    """

    orig_repo = tmpdir.mkdir('orig')

    # Ensure ah_bootstrap is imported from the local directory
    import ah_bootstrap

    with orig_repo.as_cwd():
        run_cmd('git', ['init'])

        # Write a test setup.py that uses ah_bootstrap; it also ensures that
        # any previous reference to astropy_helpers is first wiped from
        # sys.modules
        orig_repo.join('setup.py').write(TEST_SETUP_PY.format(args=''))
        run_cmd('git', ['add', 'setup.py'])

        # Add our own clone of the astropy_helpers repo as a submodule named
        # astropy_helpers
        run_cmd('git', ['submodule', 'add', str(testpackage),
                        '_astropy_helpers_test_'])

        run_cmd('git', ['commit', '-m', 'test repository'])

        os.chdir(str(tmpdir))

        # Creates a clone of our test repo in the directory 'clone'
        run_cmd('git', ['clone', 'orig', 'clone'])

        os.chdir('clone')

        run_setup('setup.py', [])

        stdout, stderr = capsys.readouterr()
        path = stdout.strip()

        # Ensure that the astropy_helpers used by the setup.py is the one that
        # was imported from git submodule
        assert path == str(tmpdir.join('clone', '_astropy_helpers_test_',
                                       '_astropy_helpers_test_',
                                       '__init__.py'))


def test_bootstrap_from_directory(tmpdir, testpackage, capsys):
    """
    Tests simply bundling a copy of the astropy_helpers source code in its
    entirety bundled directly in the source package and not in an archive.
    """

    import ah_bootstrap

    source = tmpdir.mkdir('source')
    testpackage.copy(source.join('_astropy_helpers_test_'))

    with source.as_cwd():
        source.join('setup.py').write(TEST_SETUP_PY.format(args=''))
        run_setup('setup.py', [])
        stdout, stderr = capsys.readouterr()

        stdout = stdout.splitlines()
        if stdout:
            path = stdout[-1].strip()
        else:
            path = ''

        # Ensure that the astropy_helpers used by the setup.py is the one that
        # was imported from git submodule
        assert path == str(source.join('_astropy_helpers_test_',
                                       '_astropy_helpers_test_',
                                       '__init__.py'))


def test_bootstrap_from_archive(tmpdir, testpackage, capsys):
    """
    Tests importing _astropy_helpers_test_ from a .tar.gz source archive
    shipped alongside the package that uses it.
    """

    orig_repo = tmpdir.mkdir('orig')

    # Ensure ah_bootstrap is imported from the local directory
    import ah_bootstrap

    # Make a source distribution of the test package
    with silence():
        run_setup(str(testpackage.join('setup.py')),
                  ['sdist', '--dist-dir=dist', '--formats=gztar'])

    dist_dir = testpackage.join('dist')
    for dist_file in dist_dir.visit('*.tar.gz'):
        dist_file.copy(orig_repo)

    with orig_repo.as_cwd():
        # Write a test setup.py that uses ah_bootstrap; it also ensures that
        # any previous reference to astropy_helpers is first wiped from
        # sys.modules
        args = 'path={0!r}'.format(os.path.basename(str(dist_file)))
        orig_repo.join('setup.py').write(TEST_SETUP_PY.format(args=args))

        run_setup('setup.py', [])

        stdout, stderr = capsys.readouterr()
        path = stdout.splitlines()[-1].strip()

        # Installation from the .tar.gz should have resulted in a .egg
        # directory that the _astropy_helpers_test_ package was imported from
        eggs = glob.glob('*.egg')
        assert eggs
        egg = orig_repo.join(eggs[0])
        assert os.path.isdir(str(egg))

        assert path == str(egg.join('_astropy_helpers_test_',
                                    '__init__.py'))


def test_download_if_needed(tmpdir, testpackage, capsys):
    """
    Tests the case where astropy_helpers was not actually included in a
    package, or is otherwise missing, and we need to "download" it.

    This does not test actually downloading from the internet--this is normally
    done through setuptools' easy_install command which can also install from a
    source archive.  From the point of view of ah_boostrap the two actions are
    equivalent, so we can just as easily simulate this by providing a setup.cfg
    giving the path to a source archive to "download" (as though it were a
    URL).
    """

    source = tmpdir.mkdir('source')

    # Ensure ah_bootstrap is imported from the local directory
    import ah_bootstrap

    # Make a source distribution of the test package
    with silence():
        run_setup(str(testpackage.join('setup.py')),
                  ['sdist', '--dist-dir=dist', '--formats=gztar'])

    dist_dir = testpackage.join('dist')

    with source.as_cwd():
        source.join('setup.py').write(TEST_SETUP_PY.format(args=''))
        source.join('setup.cfg').write(textwrap.dedent("""\
            [easy_install]
            find_links = {find_links}
        """.format(find_links=str(dist_dir))))

        run_setup('setup.py', [])

        stdout, stderr = capsys.readouterr()

        # Just take the last line--on Python 2.6 distutils logs warning
        # messages to stdout instead of stderr, causing them to be mixed up
        # with our expected output
        path = stdout.splitlines()[-1].strip()

        # easy_install should have worked by 'installing' astropy_helpers as a
        # .egg in the current directory
        eggs = glob.glob('*.egg')
        assert eggs
        egg = source.join(eggs[0])
        assert os.path.isdir(str(egg))

        assert path == str(egg.join('_astropy_helpers_test_',
                                    '__init__.py'))


def test_upgrade(tmpdir, capsys):
    # Run the testpackage fixture manually, since we use it multiple times in
    # this test to make different versions of _astropy_helpers_test_
    orig_dir = testpackage(tmpdir.mkdir('orig'))

    # Make a test package that uses _astropy_helpers_test_
    source = tmpdir.mkdir('source')
    dist_dir = source.mkdir('dists')
    orig_dir.copy(source.join('_astropy_helpers_test_'))

    with source.as_cwd():
        setup_py = TEST_SETUP_PY.format(args='auto_upgrade=True')
        source.join('setup.py').write(setup_py)

        # This will be used to later to fake downloading the upgrade package
        source.join('setup.cfg').write(textwrap.dedent("""\
            [easy_install]
            find_links = {find_links}
        """.format(find_links=str(dist_dir))))

    # Make additional "upgrade" versions of the _astropy_helpers_test_
    # package--one of them is version 0.2 and the other is version 0.1.1.  The
    # auto-upgrade should ignore version 0.2 but use version 0.1.1.
    upgrade_dir_1 = testpackage(tmpdir.mkdir('upgrade_1'), version='0.2')
    upgrade_dir_2 = testpackage(tmpdir.mkdir('upgrade_2'), version='0.1.1')

    dists = []
    # For each upgrade package go ahead and build a source distribution of it
    # and copy that source distribution to a dist directory we'll use later to
    # simulate a 'download'
    for upgrade_dir in [upgrade_dir_1, upgrade_dir_2]:
        with silence():
            run_setup(str(upgrade_dir.join('setup.py')),
                      ['sdist', '--dist-dir=dist', '--formats=gztar'])
        dists.append(str(upgrade_dir.join('dist')))
        for dist_file in upgrade_dir.visit('*.tar.gz'):
            dist_file.copy(source.join('dists'))

    # Monkey with the PackageIndex in ah_bootstrap so that it is initialized
    # with the test upgrade packages, and so that it does not actually go out
    # to the internet to look for anything
    import ah_bootstrap

    class FakePackageIndex(PackageIndex):
        def __init__(self, *args, **kwargs):
            PackageIndex.__init__(self, *args, **kwargs)
            self.to_scan = dists

        def find_packages(self, requirement):
            # no-op
            pass

    ah_bootstrap.PackageIndex = FakePackageIndex

    try:
        with source.as_cwd():
            # Now run the source setup.py; this test is similar to
            # test_download_if_needed, but we explicitly check that the correct
            # *version* of _astropy_helpers_test_ was used
            run_setup('setup.py', [])

            stdout, stderr = capsys.readouterr()
            path = stdout.splitlines()[-1].strip()
            eggs = glob.glob('*.egg')
            assert eggs

            egg = source.join(eggs[0])
            assert os.path.isdir(str(egg))
            assert path == str(egg.join('_astropy_helpers_test_',
                                        '__init__.py'))
            assert 'astropy_helpers_test-0.1.1-' in str(egg)
    finally:
        ah_bootstrap.PackageIndex = PackageIndex
