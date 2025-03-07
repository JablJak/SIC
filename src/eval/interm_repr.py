class IntermediateRepresentation:
    def __init__(self):
        self.single_input_size = None
        self.input_size = None
        self.input_num_el = None
        self.input_element_size = None
        self.input_shape = None
        self.input_values = None
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

        self.input_values = input[0].detach().cpu().numpy()
        self.input_shape = input[0].shape
        self.input_element_size = input[0].element_size()
        self.input_num_el = input[0].numel()
        self.input_size = self.input_element_size * self.input_num_el
        self.single_input_size = self.input_size // self.input_shape[0]

    def __str__(self):
        return (f"IntermediateRepresentation(single_input_size={self.single_input_size}, "
                f"input_size={self.input_size}, "
                f"input_num_el={self.input_num_el}, "
                f"input_element_size={self.input_element_size}, "
                f"input_shape={self.input_shape}, "
                f"single_output_size={self.single_output_size}, "
                f"shape={self.shape}, "
                f"element_size={self.element_size}, "
                f"num_el={self.num_el}, "
                f"size={self.size}")
