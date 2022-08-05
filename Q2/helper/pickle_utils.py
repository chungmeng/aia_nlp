import pickle
import gzip

#Save Pickle File
def save_pickle(obj,fullpath, compress=False, protocol=-1):
    """
    Save Pickle Object
    Args:
        obj: Object
        fullpath: Full destination to Save pickled object
        compress: Gzip [ True / False (default)]
        protocol: Gzip protocol (default = -1)
    Returns:
        None
    """ 
    if compress:
        with gzip.open(fullpath, 'wb') as f:
            pickle.dump(obj, f, protocol)
        print('Pickled Gzip object to', fullpath)
    else:
        with open(fullpath, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Pickled object to', fullpath)

#Load Pickle File
def load_pickle(fullpath, compress=False):
    """
    Load Pickled Object
    Args:
        fullpath: Full source path to Pickled object
        compress: Gzip [ True / False (default)]
    Returns:
        obj: Unpickled Object
    """ 
    if compress:
        print('Loading Pickled Gzip object from', fullpath)
        with gzip.open(fullpath, 'rb') as handle:
            return pickle.load(handle)
    else:
        print('Loading Pickled object from', fullpath)
        with open(fullpath, 'rb') as handle:
            return pickle.load(handle)

