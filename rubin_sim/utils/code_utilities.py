__all__ = ("sims_clean_up",)

import gc
import numbers

import numpy as np


def sims_clean_up():
    """
    This method will clean up data caches created by the sims software stack.
    Any time a cache is added to the sims software stack, it can be added to
    the list sims_clean_up.targets.  When sims_clean_up() is called, it will
    loop through the contents of sims_clean_up.targets.  It will call pop()
    on all of the contents of each sims_clean_up.target, run close() on each
    item it pops (if applicable), delete each item it pops, and then reset
    each sims_clean_up.target to either a blank dict or list (depending on
    what the target was).  sims_clean_up() will then run the garbage
    collector.

    Note: if a target cache is not a dict or a list, it will attempt to call
    close() on the cache and delete the cache.
    """

    if not hasattr(sims_clean_up, "targets"):
        return None

    for target in sims_clean_up.targets:
        if isinstance(target, dict):
            while len(target) > 0:
                obj = target.popitem()
                if hasattr(obj[1], "close"):
                    try:
                        obj[1].close()
                    except:
                        pass
                del obj
        elif isinstance(target, list):
            while len(target) > 0:
                obj = target.pop()
                if hasattr(obj, "close"):
                    try:
                        obj.close()
                    except:
                        pass
                del obj
        else:
            if hasattr(target, "close"):
                target.close()
            del target

    gc.collect()
    return None


sims_clean_up.targets = []


def _validate_inputs(input_list, input_names, method_name):
    """
    This method will validate the inputs of other methods.

    input_list is a list of the inputs passed to a method.

    input_name is a list of the variable names associated with
    input_list

    method_name is the name of the method whose input is being validated.

    _validate_inputs will verify that all of the inputs in input_list are:

    1) of the same type
    2) either numpy arrays or instances of numbers.Number (floats or ints)
    3) if they are numpy arrays, they all have the same length

    If any of these criteria are violated, a RuntimeError will be raised

    returns True if the inputs are numpy arrays; False if not
    """

    if isinstance(input_list[0], np.ndarray):
        desired_type = np.ndarray
    elif isinstance(input_list[0], numbers.Number):
        desired_type = numbers.Number
    else:
        raise RuntimeError(
            "The arg %s input to method %s " % (input_names[0], method_name)
            + "should be either a number or a numpy array"
        )

    valid_type = True
    bad_names = []
    for ii, nn in zip(input_list, input_names):
        if not isinstance(ii, desired_type):
            valid_type = False
            bad_names.append(nn)

    if not valid_type:
        msg = "The input arguments:\n"
        for nn in bad_names:
            msg += "%s,\n" % nn
        msg += "passed to %s " % method_name
        msg += "need to be either numbers or numpy arrays\n"
        msg += "and the same type as the argument %s" % input_names[0]
        msg += "\n\nTypes of arguments are:\n"
        for name, arg in zip(input_names, input_list):
            msg += "%s: %s\n" % (name, type(arg))
        raise RuntimeError(msg)

    if desired_type is np.ndarray:
        same_length = True
        for ii in input_list:
            if len(ii) != len(input_list[0]):
                same_length = False
        if not same_length:
            raise RuntimeError("The arrays input to %s " % method_name + "all need to have the same length")

    if desired_type is np.ndarray:
        return True

    return False
