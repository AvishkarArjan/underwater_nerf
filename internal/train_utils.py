import numpy as np
from internal import utils
from internal import models

from torch.utils._pytree import tree_map, tree_flatten

def create_optimizer():
    pass

def create_render_fn():
    pass

def create_train_step():
    pass

#
def tree_sum(tree):
    pass
# multinerf-pytorch
def tree_len(tree:list):
    return tree_sum(tree_map(lambda z: np.prod(z.shape), tree))

def setup_model(
        config: configs.Config,
        rng: np.ndarray,
        dataset: Optional[datasets.Dataset] = None
        ):
    """
    Return type
        
    """
    
    dummy_rays = utils.dummy_rays()
    model, variables = models.construct_model(rng, dummy_rays, config)

    state, lr_fn = create_optimizer(config, variables)
    render_eval_pfn = create_render_fn(model)
    train_pstep = create_train_step(model, config, dataset=dataset)

    return model, state, render_eval_pfn, train_pstep, lr_fn
    