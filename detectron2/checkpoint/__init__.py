import torch
import os

class DetectionCheckpointer:
    def __init__(self, model, save_dir=""):
        self.model = model
        self.save_dir = save_dir
        
    def load(self, path):
        if not path:
             print("[MockD2] Invalid path provided.")
             return {}
             
        if not os.path.exists(path):
            print(f"[MockD2] Weight file not found: {path}")
            return {}
            
        print(f"[MockD2] Loading weights from {path}...")
        try:
            # PyTorch 2.6 default is weights_only=True, which blocks Omegaconf objects in D2 checkpoints.
            # We explicitly set weights_only=False to allow loading the trusted checkpoint.
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            # Detectron2 usually saves weights in 'model' key
            # In some cases, it might be the root dict or 'state_dict'
            state_dict = checkpoint.get("model", checkpoint)
            if "state_dict" in checkpoint and not "model" in checkpoint:
                state_dict = checkpoint["state_dict"]
            
            # Remove possible 'module.' prefix if weights were saved from DataParallel
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"[MockD2] Weight loading result: {msg}")
            return checkpoint
        except Exception as e:
            print(f"[MockD2] Failed to load checkpoint: {e}")
            return {}
