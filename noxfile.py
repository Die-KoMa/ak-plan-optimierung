"""This module implements our CI function calls."""

import nox


def _setup_test_session(session):
    # install pip via conda to handle pip deps
    session.conda_install("pip", channel="conda-forge")
    # install core solvers
    session.conda_install("highs", channel="conda-forge")
    session.conda_install("gurobi", channel="gurobi")
    # install project
    session.conda_install("numpy", channel="conda-forge")
    session.conda_install("dacite", channel="conda-forge")
    session.install(".")
    session.install("pytest")
    session.install("pytest-timeout")
    return session


@nox.session(name="test", venv_backend="mamba")
def run_test(session):
    """Run pytest on all test cases besides the extensive suite."""
    session = _setup_test_session(session)
    session.run("pytest", "-m", "not extensive", *session.posargs)


@nox.session(name="fast-test", venv_backend="mamba")
def run_test_fast(session):
    """Run pytest on fast test cases."""
    session = _setup_test_session(session)
    session.run("pytest", "-m", "not slow and not extensive", *session.posargs)

@nox.session(name="extensive-test", venv_backend="mamba")
def run_test_extensive(session):
    """Run pytest on all test cases."""
    session = _setup_test_session(session)
    session.run("pytest", *session.posargs)


@nox.session(name="lint")
def lint(session):
    """Check code conventions."""
    session.install("flake8")
    session.install(
        "flake8-colors",
        "flake8-black",
        "flake8-docstrings",
        "flake8-bugbear",
        "flake8-broken-line",
        "pep8-naming",
        "pydocstyle",
        "darglint",
    )
    session.run("flake8", "src", "tests", "noxfile.py", *session.posargs)


@nox.session(name="typing")
def mypy(session):
    """Check type hints."""
    session.install(".")
    session.install("mypy")
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--ignore-missing-imports",
        "--strict",
        "src",
        *session.posargs
    )


@nox.session(name="format")
def format(session):
    """Fix common convention problems automatically."""
    session.install("black")
    session.install("isort")
    session.run("isort", "src", "tests", "noxfile.py")
    session.run("black", "src", "tests", "noxfile.py")


@nox.session(name="coverage")
def check_coverage(session):
    """Check test coverage and generate a html report."""
    session.install(".")
    session.install("pytest")
    session.install("pytest-timeout")
    session.install("coverage")
    try:
        session.run("coverage", "run", "-m", "pytest", *session.posargs)
    finally:
        session.run("coverage", "html")


@nox.session(name="coverage-clean")
def clean_coverage(session):
    """Remove the code coverage website."""
    session.run("rm", "-r", "htmlcov", external=True)
