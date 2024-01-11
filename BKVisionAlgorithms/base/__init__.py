from pypattyrn.creational.singleton import Singleton


def register():
    """
    Register the name of the class
    Args:
        name: name of the class
    Returns:

    """
    model = SingModelAll()

    def inner(cls):
        for name in cls.names:
            model.register(name, cls)
        return cls

    return inner


class SingModelAll(metaclass=Singleton):
    """
    DetectionInterface is an abstract class that defines the interface for
    detection algorithms. All detection algorithms should inherit from this
    class.
    """

    def __init__(self):
        """
        Initialize the DetectionInterface object.
        """
        self.models = {}
        super().__init__()

    def register(self, name, class_):
        self.models[name.lower()] = class_

    def create(self, property_):
        print(property_.framework)
        if property_.framework:
            return self.models[property_.framework.lower() + "_frame"](property_)
        return self.models[property_.name.lower()](property_)

    def get_model_list(self):
        res = {}
        for key in self.models.keys():
            res[key] = self.models[key].get_model_list()
        return res
