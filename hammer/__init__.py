try:
    """
    Try to load the hammer version from the package metadata to avoid using static version strings.
    This allows for dynamic versioning based on the installed package version.
    """
    from importlib.metadata import version

    __version__ = version("hammer")
except Exception:
    """
    Package metadata is not avaiable when only the hammer module is used on a ray worker.
    When a ray job is submitted, the dynamically loaded version above is stored in the environment variable HAMMER_VERSION.
    This fallback ensures that the version is still accessible in such cases.
    """
    import os

    __version__ = os.getenv("HAMMER_VERSION", "0.0.0")
