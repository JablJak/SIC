class IntermediateRepresentation:
    def __init__(self):
        self.single_output_size = None
        self.values = None
        self.shape = None
        self.element_size = None
        self.num_el = None
        self.size = None

    def hook_fn(self, model, input, output):
        self.values = output.detach().cpu().numpy()
        self.shape = output.shape
        self.element_size = output.element_size()
        self.num_el = output.numel()
        self.size = self.element_size * self.num_el
        self.single_output_size = self.size // self.shape[0]

    def __str__(self):
        return (f"IntermediateRepresentation(shape={self.shape}, size={self.size}, "
                f"single_output_size={self.single_output_size})")