import torch


def to_cuda_if_available(*tensors):
    if torch.cuda.is_available():
        tensors = [tensor.cuda() if tensor is not None else None for tensor in tensors]
    if len(tensors) == 1:
        return tensors[0]
    return tensors


def to_cpu_if_available(*tensors):
    if torch.cuda.is_available():
        tensors = [tensor.cpu() if tensor is not None else None for tensor in tensors]
    if len(tensors) == 1:
        return tensors[0]
    return tensors


def load_without_cuda(model, state_dict_path):
    model.load_state_dict(torch.load(state_dict_path, map_location=lambda storage, loc: storage))
