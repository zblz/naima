from pkg_resources import DistributionNotFound, get_distribution


def get_version():
    try:
        return get_distribution("naima").version
    except DistributionNotFound:
        # package is not installed
        return "UNINSTALLED"
