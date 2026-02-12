from detectron2.config import ConfigDict

def get_config(name):
    # DiffMatte uses this to get AdamW from optim.py
    # We return a mock that satisfies the attribute chain
    res = ConfigDict()
    res.AdamW = ConfigDict()
    res.AdamW.params = ConfigDict()
    return res
