from torch.utils import _pytree as pytree


class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        tensors = self.tensors.to(*args, **kwargs)
        mask = None
        if self.mask is not None:
            mask = self.mask.to(
                device=tensors.device,
                non_blocking=kwargs.get("non_blocking", False),
                copy=kwargs.get("copy", False),
            )
        return type(self)(tensors, mask)

    def clone(self):
        tensors = self.tensors.clone()
        mask = None if self.mask is None else self.mask.clone()
        return type(self)(tensors, mask)

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

    @property
    def device(self):
        return self.tensors.device

    @property
    def shape(self):
        return self.tensors.shape

    def pin_memory(self, device=None):
        self.tensors = self.tensors.pin_memory(device)
        if self.mask is not None:
            self.mask = self.mask.pin_memory(device)
        return self


pytree.register_pytree_node(
    NestedTensor,
    lambda value: ([value.tensors, value.mask], None),
    lambda values, _: NestedTensor(values[0], values[1]),
)
