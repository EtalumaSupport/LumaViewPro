
import enum


@enum.unique
class PostFunction(enum.Enum):
    COMPOSITE = "Composite"
    STITCHED = "Stitched"
    ZPROJECT = "ZProject"
    VIDEO = "Video"
    STACK = "Stack"

    @classmethod
    def list_values(cls):
        return list(map(lambda c: c.value, cls))
