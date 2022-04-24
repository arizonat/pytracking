from pytracking.evaluation import Tracker, get_dataset, trackerlist


def pysot_fish_test():
    trackers = trackerlist('pysot', 'siammask', range(5)) + \
               trackerlist('pysot', 'dasiamrpn', range(5)) + \
               trackerlist('pysot', 'siamrpnpp', range(5))
    dataset = get_dataset('fish')
    return trackers, dataset

def fish_test():
    trackers = trackerlist('keep_track', 'default', range(5)) + \
               trackerlist('dimp', 'super_dimp', range(5)) + \
               trackerlist('atom', 'default', range(5))
    dataset = get_dataset('fish')
    return trackers, dataset

def older_fish_test():
    trackers = trackerlist('kys', 'default', range(5)) + \
               trackerlist('dimp', 'dimp50', range(5)) + \
               trackerlist('eco', 'default', range(5))
    dataset = get_dataset('fish')
    return trackers, dataset
    

def debug_fish_test():
    trackers = trackerlist('pysot', 'dasiamrpn', range(1))
    dataset = get_dataset('fish')

    return trackers, dataset
