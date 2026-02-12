import torch

class ImageList:
    def __init__(self, tensor, image_sizes):
        self.tensor = tensor
        self.image_sizes = image_sizes
        
    def __len__(self):
        return len(self.tensor)
        
    def __getitem__(self, idx):
        return self.tensor[idx]

    @staticmethod
    def from_tensors(tensors, size_divisibility=0, pad_value=0.0):
        # Simplistic mock: assume tensors are already of same size or just a single batch tensor
        if isinstance(tensors, torch.Tensor):
            return ImageList(tensors, [tensors.shape[-2:]] * tensors.shape[0])
        
        # If it's a list of tensors, we should theoretically pad them
        # But for DiffMatte inference, we only pass a single batch via wrapper
        return ImageList(tensors[0] if isinstance(tensors, list) else tensors, [])
