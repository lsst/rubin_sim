# Eventually we want to merge this into a single file with the slicers
# Max is developing. We are keeping them separate for now.

__all__ = ("LynxSamplerSlicer",)

import numpy as np

from .user_points_slicer import UserPointsSlicer


class LynxSamplerSlicer(UserPointsSlicer):
    """A spatial slicer based on samples from LightCurveLynx.

    Each 'slice' corresponds to a sample from LightCurveLynx, defined
    by its RA and Dec coordinates. It contains a reference to the model
    and a GraphState object with all of the parameter information.

    Parameters
    ----------
    samples : `lightcurvelynx.graph_state.GraphState` or int
        User-provided set of samples from LightCurveLynx (GraphState object)
        or the number of samples to generate from the model.
    model : `lightcurvelynx.models.physical_model.BasePhysicalModel`
        User-provided physical model from LightCurveLynx. This will be used
        to drive the simulations.
    **kwargs
        Additional keyword arguments passed to the parent class.
    """

    def __init__(self, samples, model, **kwargs):
        try:
            from lightcurvelynx.graph_state import GraphState
        except ImportError:
            raise ImportError(
                "LightCurveLynx is needed to run the LynxSamplerSlicer. "
                "It is not installed by default. Install it with "
                "`pip install lightcurvelynx`."
            )

        # If we are not given a predefined set of samples, generate them now.
        if isinstance(samples, int):
            num_samples = samples
            samples = model.sample_parameters(num_samples=num_samples)
        elif isinstance(samples, GraphState):
            num_samples = samples.num_samples
        else:
            raise ValueError("samples must be an int or a GraphState instance.")

        # Extract the RA and Dec coordinates from the model for each sample
        # and use them to set up the UserPointsSlicer.
        ra = np.atleast_1d(model.get_param(samples, "ra"))
        dec = np.atleast_1d(model.get_param(samples, "dec"))
        super().__init__(ra=ra, dec=dec, **kwargs)

        # Save the model to the slicer once.
        self.slice_points["lynx_model"] = model

        # Save each individual sample. Because of how numpy would treat a list
        # of dicts, we need to preallocate the numpy array with dtype=object
        # and add the states one at a time.
        state_array = np.empty(num_samples, dtype=object)
        for i, state in enumerate(samples):
            state_array[i] = state
        self.slice_points["lynx_params"] = state_array
