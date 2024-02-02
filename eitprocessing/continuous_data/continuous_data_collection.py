from . import ContinuousData


class ContinuousDataCollection(dict):

    def __setitem__(self, __key: str, __value: ContinuousData) -> None:
        self._check_item(__value, key=__key)
        return super().__setitem__(__key, __value)

    def add(self, *item: ContinuousData, overwrite: bool = False) -> None:
        for item_ in item:
            self._check_item(item_, overwrite=overwrite)
            super().__setitem__(item_.name, item_)

    def _check_item(
        self, item: ContinuousData, key=None, overwrite: bool = False
    ) -> None:
        if not isinstance(item, ContinuousData):
            raise TypeError(f"type of `data` is {type(item)}, not 'ContinuousData'")

        if key and key != item.name:
            raise KeyError(f"'{key}' does not match variant name '{item.name}'.")

        if not overwrite and key in self:
            raise KeyError(
                f"Variant with name {key} already exists. Use `overwrite=True` to overwrite."
            )
