"""

FusionGalleryConstraint represents a sketch constraint in the Fusion 360 Gallery format

"""

import uuid

class FusionGalleryConstraint:
    def __init__(self, constraint, points, curves, entity_map):
        """
        Intialize a FusionGalleryConstraint

        Args
            constraint (dict): Constraint in the deepmind format, 
                            i.e. sketch["constraintSequence"]["constraints"][n]
                            which contains for example:
                            {
                                "coincidentConstraint": {
                                    "entities": [
                                        1,
                                        7
                                    ]
                                }
                            }
            points(dict): Dictionary of FG points, where keys are uuids and values
                            are FG point dicts
            curves (dict): Dictionary of FG curves, where keys are uuids and values
                            are FG curve dicts
            entity_map (dict): Dictionary where the keys are the original numeric
                            index of the entity in the deepmind data, and the
                            values contain a dict with a type and uuid. This map
                            allows us to connect the deepmind indices to the 
                            unique uuids used in the FG dicts
        """        
        self.points = points
        self.curves = curves
        self.entity_map = entity_map
        # UUID of the constraint
        self.uuid = str(uuid.uuid1())
        # Type of constraint in deepmind terms
        self.type = list(constraint.keys())[0]
        # The entities referenced in the deepmind data
        # stored in order of their index, in FG dict format
        # with the addition of a uuid key
        self.entities = []
        # Count the number of valid entities
        self.entity_count = 0
        for cst_ent_index in constraint[self.type]["entities"]:
            # Check that the referenced entity actually exists
            if cst_ent_index not in entity_map:
                self.entities.append(None)
                continue
            # Type is either point or curve
            cst_type = entity_map[cst_ent_index]["type"]
            # UUID into either the points or curves dicts
            cst_uuid = entity_map[cst_ent_index]["uuid"]
            if cst_type == "point":
                ent = points[cst_uuid]
            elif cst_type == "curve":
                ent = curves[cst_uuid]
            # New dictionary with the additional uuid value
            ent_data = {
                **ent,
                "uuid": cst_uuid
            }
            self.entities.append(ent_data)
            self.entity_count += 1
    
    def to_dict(self):
        """Make a Fusion 360 Gallery format dict for the constraint"""
        constraint_dict = None
        if self.type == "coincidentConstraint":
            constraint_dict = self.make_coincident_constraint_cases()
        elif self.type == "horizontalConstraint":
            constraint_dict = self.make_horizontal_constraint_dict()
        elif self.type == "parallelConstraint":
            constraint_dict = self.make_parallel_constraint_dict()
        elif self.type == "verticalConstraint":
            constraint_dict = self.make_vertical_constraint_dict()
        elif self.type == "tangentConstraint":
            constraint_dict = self.make_tangent_constraint_dict()
        elif self.type == "perpendicularConstraint":
            constraint_dict = self.make_perpendicular_constraint_dict()
        elif self.type == "midpointConstraint":
            constraint_dict = self.make_midpoint_constraint_dict()
        elif self.type == "equalConstraint":
            constraint_dict = self.make_equal_constraint_dict()
        elif self.type == "concentricConstraint":
            constraint_dict = self.make_concentric_constraint_dict()

        return constraint_dict

    def type_for_entity(self, index):
        ent_type = self.entities[index]["type"]
        if ent_type == "Point3D":
            return "SketchPoint"
        return ent_type

    def is_entity_type(self, index, entity_type):
        if self.entities[index]["type"] == entity_type:
            return True
        return False
    
    def is_entity_point(self, index):
        return self.is_entity_type(index, "Point3D")

    def entity_points_identical(self):
        first_uuid = self.entities[0]["uuid"]
        for ent in self.entities:
            if ent["type"] != "Point3D":
                return False
            if ent["uuid"] != first_uuid:
                return False
        return True

    def is_entity_line(self, index):
        return self.is_entity_type(index, "SketchLine")

    def is_entity_arc(self, index):
        return self.is_entity_type(index, "SketchArc")

    def is_entity_circle(self, index):
        return self.is_entity_type(index, "SketchCircle")

    def make_coincident_constraint_cases(self):

        # assert self.entity_count == 2
        if self.is_entity_line(0) and self.is_entity_line(1):
            return self.make_collinear_constraint_dict()
        if self.is_entity_point(0) or self.is_entity_point(1):
            # In the OnShape world geometric entities own their points.
            # For example a line owns its start and end point.
            # Coincident constraints are then used to hold the end points together.

            # In the Fusion Gallery world things are a little different.
            # Fusion Gallery has SketchPoints which are geometric entities all by themselves.
            # Geometric entities like lines will then reference the point entities
            # and conicindent constraints are not required to hold them together.

            # Basically this means we need to remove some points and coincident constraints
            # from the data.

            # Check if both entities refer to the same merged point to skip 
            # if self.entities[0]["uuid"] == self.entities[1]["uuid"]:
            if self.entity_points_identical():
                return None
            return self.make_coincident_constraint_dict()
        if self.is_entity_arc(0) or self.is_entity_arc(1):
            print("Warning - SketchArc, SketchArc coincident constraint not supported")
            return None
        if self.is_entity_circle(0) or self.is_entity_circle(1):
            print("Warning - SketchCircle, SketchCircle coincident constraint not supported")
            return None
        assert False, "Unknown case"

    def make_coincident_constraint_dict(self):
        """
        {
            "type": "CoincidentConstraint",
            "entity": "301ed434-b47f-11ea-8e78-180373af3277",
            "point": "30278562-b47f-11ea-be85-180373af3277"
        }
        """
        point = None
        entity = None
        if self.is_entity_point(0):
            point = self.entities[0]
            entity = self.entities[1]
        elif self.is_entity_point(1):
            point = self.entities[1]
            entity = self.entities[0]
        if point is None:
            # Unhandled case.  Coincident lines.  Should this be a colinear constraint?
            # {
            # "type": "CollinearConstraint",
            # "line_one": "d7852898-b74c-11ea-b79f-180373af3277",
            # "line_two": "d7854f9e-b74c-11ea-9502-180373af3277"
            # }
            assert False, "Implement collinear"
            return
        return {
            "type": "CoincidentConstraint",
            "entity": entity["uuid"],
            "point": point["uuid"]
        }

    def make_collinear_constraint_dict(self):
        """
        {
        "type": "CollinearConstraint",
        "line_one": "d7852898-b74c-11ea-b79f-180373af3277",
        "line_two": "d7854f9e-b74c-11ea-9502-180373af3277"
        }
        """
        return {
            "type": "CollinearConstraint",
            "line_one": self.entities[0]["uuid"],
            "line_two": self.entities[1]["uuid"]
        }
    
    def make_horizontal_constraint_dict(self):
        """
        {
            "line": "35d8a464-e0c6-11ea-9b94-c85b76a75ed8",
            "type": "HorizontalConstraint"
        }
        """
        if self.entity_count == 2:
            if self.is_entity_point(0) and self.is_entity_point(1):
                return {
                    "point_one": self.entities[0]["uuid"],
                    "point_two": self.entities[1]["uuid"],
                    "type": "VerticalPointsConstraint"
                }
        elif self.is_entity_line(0):
            return {
                "line": self.entities[0]["uuid"],
                "type": "HorizontalConstraint"
            }
        else:
            assert False, "Unknown horizontal contraint entities"
        return None
    
    def make_parallel_constraint_dict(self, cst):
        """
        {
            "type": "ParallelConstraint",
            "line_one": "44fae288-b820-11ea-9c62-180373af3277",
            "line_two": "44fc8ff4-b820-11ea-a78c-180373af3277"
        }
        """
        assert self.is_entity_line(0)
        assert self.is_entity_line(1)
        return {
            "type": "ParallelConstraint",
            "line_one": self.entities[0]["uuid"],
            "line_two": self.entities[1]["uuid"]
        }