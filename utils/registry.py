# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py

class Registry:
    """
    The Registry that provides name (str) -> object mapping, to support third-party
    users' custom modules.

    To create a Registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = dict()

    def _do_registry(self, name, obj):
        assert (name not in self._obj_map), f"An object named '{name}' was already registered in {self._name} registry"

        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        Register the given object under the the name `obj.__name__`.
        """
        # used as a decorator
        if obj is None:
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_registry(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_registry(name, obj)

    def get(self, name):
        obj = self._obj_map.get(name)
        if obj is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return obj

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


# define registries
DATASET_REGISTRY = Registry('dataset')
NETWORK_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')