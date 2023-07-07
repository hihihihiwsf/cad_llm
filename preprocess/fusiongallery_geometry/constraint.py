"""

FusionGalleryConstraint represents a sketch constraint in the Fusion 360 Gallery format

"""

import uuid

from preprocess.fusiongallery_geometry.base_constraint import FusionGalleryBaseConstraint
from preprocess.fusiongallery_geometry.line import FusionGalleryLine


class FusionGalleryConstraint(FusionGalleryBaseConstraint):
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
        super().__init__(constraint, points, curves, entity_map)
    
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

    def make_coincident_constraint_cases(self):
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
        if self.entity_count == 2 and self.is_entity_point(0) and self.is_entity_point(1):
            return {
                "point_one": self.entities[0]["uuid"],
                "point_two": self.entities[1]["uuid"],
                "type": "HorizontalPointsConstraint"
            }
        elif self.entity_count == 1 and self.is_entity_line(0):
            return {
                "line": self.entities[0]["uuid"],
                "type": "HorizontalConstraint"
            }
        elif self.entity_count > 2 and self.are_entities_lines():
            # Handle multiple separate constraints
            multi_cst = []
            for ent in self.entities:
                multi_cst.append({
                    "line": ent["uuid"],
                    "type": "HorizontalConstraint"
                })
            return multi_cst
        else:
            assert False, "Unknown horizontal contraint entities"
    
    def make_parallel_constraint_dict(self):
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

    def make_vertical_constraint_dict(self):
        """
        {
            "line": "35512506-e0c6-11ea-88c8-c85b76a75ed8",
            "type": "VerticalConstraint"
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
                "type": "VerticalConstraint"
            }
        else:
            assert False, "Unknown vertical constraint entities"
        return None
    
    def make_tangent_constraint_dict(self):
        """
        {
            "curve_one": "35528476-e0c6-11ea-95eb-c85b76a75ed8",
            "curve_two": "3551c13b-e0c6-11ea-bfd3-c85b76a75ed8",
            "type": "TangentConstraint"
        }
        """
        assert self.is_entity_curve(0)
        assert self.is_entity_curve(0)
        return {
            "curve_one": self.entities[0]["uuid"], # first
            "curve_two": self.entities[1]["uuid"], # second
            "type": "TangentConstraint"
        }

    def make_perpendicular_constraint_dict(self):
        """
        {
            "line_one": "35520f42-e0c6-11ea-a553-c85b76a75ed8",
            "line_two": "35512506-e0c6-11ea-88c8-c85b76a75ed8",
            "type": "PerpendicularConstraint"
        
        """
        assert self.is_entity_line(0)
        assert self.is_entity_line(1)
        return {
            "line_one": self.entities[0]["uuid"], # first
            "line_two": self.entities[1]["uuid"], # second
            "type": "PerpendicularConstraint"
    }

    def make_midpoint_constraint_dict(self):
        """
        {
            "type": "MidPointConstraint",
            "point": "ea5239a8-b406-11ea-91aa-180373af3277",
            "mid_point_curve": "ea4bab00-b406-11ea-8100-180373af3277"
        }
        """
        if self.entity_count == 2:
            # Two entity case maps directly to what the FG format provides, i.e.:
            # - point	Returns the sketch point being constrained.
            # - midPointCurve	Returns the curve defining the midpoint.
            
            # {
            #     "midpointConstraint": {
            #         "midpoint": 14,
            #         "entity": 4
            #     }
            # }            
            assert self.is_entity_point(0)
            assert self.is_entity_curve(1)

            return {
                "type": "MidPointConstraint",
                "point": self.entities[0]["uuid"],          # midpoint
                "mid_point_curve": self.entities[1]["uuid"] # entity
            }
        elif self.entity_count == 3:
            # Three entity case we need to handle in a special way...
            # 
            # {
            #     "midpointConstraint": {
            #         "midpoint": 12,
            #         "endpoints": {
            #             "first": 5,
            #             "second": 1
            #         }
            #     }
            # }
            # 
            # We create a sketch line, set it as construction geometry, 
            # then make the mid point the middle of the line
            start_point = self.entities[1]["uuid"]
            end_point = self.entities[2]["uuid"]
            fg_line = FusionGalleryLine.create_manual_line_dict(start_point, end_point, True)
            fg_line_uuid = str(uuid.uuid1())
            self.curves[fg_line_uuid] = fg_line

            return {
                "type": "MidPointConstraint",
                "point": self.entities[0]["uuid"],  # midpoint
                "mid_point_curve": fg_line_uuid     # endpoints line
            }
        else:
            assert False, "Unknown midpoint constraint entities"

    def make_equal_constraint_dict(self):
        """
        {
            "curve_one": "3551e838-e0c6-11ea-a5b3-c85b76a75ed8",
            "curve_two": "3551c13b-e0c6-11ea-bfd3-c85b76a75ed8",
            "type": "EqualConstraint"
        }
        """
        assert self.is_entity_curve(0)
        assert self.is_entity_curve(1)

        if self.entity_count == 2:
            return {
                "curve_one": self.entities[0]["uuid"],
                "curve_two": self.entities[1]["uuid"],
                "type": "EqualConstraint"
            }

        elif self.entity_count > 2:
            # Loop to handle multiple entity equal constraints
            # e.g.
            # 
            # "equalConstraint": {
            #     "entities": [
            #         0,
            #         2,
            #         4,
            #         6,
            #         14,
            #         12,
            #         10,
            #         8
            #     ]
            # }
            multi_cst = []
            for index in range(self.entity_count - 1):
                multi_cst.append({
                    "curve_one": self.entities[index]["uuid"],
                    "curve_two": self.entities[index + 1]["uuid"],
                    "type": "EqualConstraint"
                })
            return multi_cst

    def make_concentric_constraint_dict(self):
        """
        {
            "curve_one": "35d940b2-e0c6-11ea-a286-c85b76a75ed8",
            "curve_two": "35d940b0-e0c6-11ea-b276-c85b76a75ed8",
            "type": "ConcentricConstraint"
        }
        """
        curve_one = self.entities[0]["uuid"]
        curve_two = self.entities[1]["uuid"]

        # Find the parent curves for points
        if self.is_entity_point(0):
            curve_one = self.entities[0]["parent"]
        if self.is_entity_point(1):
            curve_two = self.entities[1]["parent"]

        return {
            "curve_one": curve_one,
            "curve_two": curve_two,
            "type": "ConcentricConstraint"
        }