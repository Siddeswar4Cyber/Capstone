"""Dashboard backend and frontend, with lazy historical API compatibility."""


def __getattr__(name):
    from importlib import import_module

    server = import_module("ppfcl_nids.dashboard.server")

    try:
        return getattr(server, name)
    except AttributeError:
        raise AttributeError(f"Dashboard has no attribute {name!r}") from None
