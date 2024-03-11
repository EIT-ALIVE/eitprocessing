from typing import NoReturn


class TimeDomainFilter:
    available_in_gui = True

    def apply_filter(self, input_data) -> NoReturn:
        msg = "Implement in subclass"
        raise NotImplementedError(msg)
