from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "dragnuwa.svd.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
