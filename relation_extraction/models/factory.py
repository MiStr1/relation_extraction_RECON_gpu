from RECON.models import baselines
from RECON.models import models


def get_model(model_name):
    """
    A factory design mode. Get model by its name.
    Input:
    model_name: a string
    Output:
    model: the model required
    """
    try:
        model = getattr(baselines, model_name)
        return model
    except:
        try:
            model = getattr(models, model_name)
            return model
        except:
            print("Model %s has not been implemented.")
            raise NotImplementedError
