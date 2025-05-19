from functools import cached_property


class CachedPropertyCollector(type):
    def __init__(cls, name: str, bases: tuple, namespace: dict):
        super().__init__(name, bases, namespace)
        cls.CACHED_PROPERTY_NAMES = [key for key, value in namespace.items() if isinstance(value, cached_property)]


class CachedPropertyInvalidator(metaclass=CachedPropertyCollector):
    def __init__(self):
        super().__init__()

    def invalidate_cached_properties(self):
        for key in self.__class__.CACHED_PROPERTY_NAMES:
            if hasattr(self, key):
                delattr(self, key)
