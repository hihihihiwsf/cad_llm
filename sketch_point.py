"""

Copied from https://git.autodesk.com/Research/SketchDL/blob/master/datagen/sketch_point.py

"""

class SketchPoint:
    """

    Class representing a point in a sketch with an index
    Used to weld curve points together

    """
    def __init__(self, x=0.0, y=0.0, z=0.0, json_data=None):
        if json_data:
            self.x = json_data["x"]
            self.y = json_data["y"]
            self.z = json_data["z"]
        else:
            self.x = x
            self.y = y
            self.z = z
        self.index = None
