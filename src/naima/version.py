from pkg_resources import get_distribution, DistributionNotFound


def get_version():
    try:
        return get_distribution("naima").version
    except DistributionNotFound:
        # package is not installed
        return "UNINSTALLED"
