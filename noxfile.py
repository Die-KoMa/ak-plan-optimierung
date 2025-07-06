"""This module implements our CI function calls."""

import nox


def _setup_test_session(session):
    session.install(".[test]")
    session.run(
        "python",
        "-c",
        "import linopy; print('Available solvers:', linopy.solvers.available_solvers)",
    )
    return session


@nox.session(name="test")
def run_test(session):
    """Run pytest on all test cases besides the extensive suite."""
    session = _setup_test_session(session)
    session.run("pytest", "-m", "not extensive", *session.posargs)


@nox.session(name="fast-test")
def run_test_fast(session):
    """Run pytest on fast test cases."""
    session = _setup_test_session(session)
    session.run("pytest", "-m", "not slow and not extensive", *session.posargs)


@nox.session(name="extensive-test")
def run_test_extensive(session):
    """Run pytest on all test cases."""
    session = _setup_test_session(session)
    session.run("pytest", *session.posargs)


@nox.session(name="lint")
def lint(session):
    """Check code conventions."""
    session.install(".[lint]")
    session.run("flake8", "src", "tests", "noxfile.py", *session.posargs)


@nox.session(name="typing")
def mypy(session):
    """Check type hints."""
    session.install(".[typing]")
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--ignore-missing-imports",
        "--strict",
        "src",
        "tests",
        *session.posargs,
    )


@nox.session(name="format")
def format(session):
    """Fix common convention problems automatically."""
    session.install(".[format]")
    session.run("isort", "src", "tests", "noxfile.py")
    session.run("black", "src", "tests", "noxfile.py")


@nox.session(name="coverage")
def check_coverage(session):
    """Check test coverage and generate a html report."""
    session.install(".[coverage,test]")
    try:
        session.run("coverage", "run", "-m", "pytest", *session.posargs)
    finally:
        session.run("coverage", "html")


@nox.session(name="coverage-clean")
def clean_coverage(session):
    """Remove the code coverage website."""
    session.run("rm", "-r", "htmlcov", external=True)
