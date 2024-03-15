class ParameterExtraction:
    available_in_gui = True

    def compute_parameter(self, sequence):
        raise NotImplementedError("Implement in subclass")
