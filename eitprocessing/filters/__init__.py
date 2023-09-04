class TimeDomainFilter:
    available_in_gui = True
    
    def apply_filter(self, input_data):
        raise NotImplementedError("Implement in subclass")
