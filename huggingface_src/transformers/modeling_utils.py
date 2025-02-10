import torch.nn as nn


class ModuleUtilsMixin:
    pass


class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin, PeftAdapterMixin):
    pass





