"""

FusionGalleryConstraint represents a sketch constraint in the Fusion 360 Gallery format

"""

import uuid

from preprocess.fusiongallery_geometry.base_constraint import FusionGalleryBaseConstraint
from preprocess.fusiongallery_geometry.line import FusionGalleryLine


class FusionGalleryConstraint(FusionGalleryBaseConstraint):

    # Supported Fusion 360 constraint types
    types = {
        "CoincidentConstraint",
        "CollinearConstraint",
        "HorizontalConstraint",
        "HorizontalPointsConstraint",
        "ParallelConstraint",
        "VerticalConstraint",
        "VerticalPointsConstraint",
        "TangentConstraint",
        "PerpendicularConstraint",
        "MidPointConstraint",
        "EqualConstraint",
        "ConcentricConstraint",
        "SymmetryConstraint"
    }
    # Currently these constraint types are not supported
    # 
    # Offset = 13   
    # Fix = 16
    # Projected = 1
    # Circular_Pattern = 18
    # Pierce = 19
    # Linear_Pattern = 20
    # Centerline_Dimension = 21
    # Intersected = 22
    # Silhoutted = 23
    # Quadrant = 24
    # Normal = 25
    # Minor_Diameter = 26
    # Major_Diameter = 27
    # Rho = 28
    # Unknown = 29
    # Subnode = 101   

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
        elif self.type == "mirrorConstraint":
            constraint_dict = self.make_symmetry_constraint_dict()
        elif self.type == "fixConstraint":
            constraint_dict = self.apply_fix_constraint()
        else:
            self.converter.log_failure(f"{self.type} constraint not supported")
            return None
        return constraint_dict

    def make_coincident_constraint_cases(self):
        if self.is_entity_line(0) and self.is_entity_line(1):
            return self.make_collinear_constraint_dict()
        
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
            # Return a special flag to indicate the points have been merged
            return "Merge"
        
        # Handle the standard case of a constraint between a point and a line
        cst_dict = self.make_coincident_constraint_dict()
        if cst_dict is not None:
            return cst_dict

        # Report the constraint entities failing
        entity_types = sorted([self.entities[0]['type'], self.entities[1]['type']])
        self.converter.log_failure(f"coincidentConstraint has unsupported entities {entity_types[0]} and {entity_types[1]}")
        return None

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
        # Fusion expects a coincident constraint between a point and a point/curve
        if self.is_entity_point(0) and self.is_entity_point(1):
            point = self.entities[0]
            entity = self.entities[1]
        elif self.is_entity_point(0) and self.is_entity_curve(1):
            point = self.entities[0]
            entity = self.entities[1]
        elif self.is_entity_point(1) and self.is_entity_curve(0):
            point = self.entities[1]
            entity = self.entities[0]
        if point is None:
            return None
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
        elif self.entity_count > 1 and self.are_entities_lines():
            # Handle multiple separate line constraints
            multi_cst = []
            for ent in self.entities:
                multi_cst.append({
                    "line": ent["uuid"],
                    "type": "HorizontalConstraint"
                })
            return multi_cst
        elif self.entity_count > 1 and self.are_entities_points():
            # Handle multiple separate point constraints
            multi_cst = []
            for index in range(self.entity_count - 1):
                multi_cst.append({
                    "point_one": self.entities[index]["uuid"],
                    "point_two": self.entities[index + 1]["uuid"],
                    "type": "HorizontalPointsConstraint"
                })
            return multi_cst
        else:
            entity_types = sorted([self.entities[0]['type'], self.entities[1]['type']])
            self.converter.log_failure(f"horizontalConstraint has unsupported entities {entity_types[0]} and {entity_types[1]}")
            return None
    
    def make_parallel_constraint_dict(self):
        """
        {
            "type": "ParallelConstraint",
            "line_one": "44fae288-b820-11ea-9c62-180373af3277",
            "line_two": "44fc8ff4-b820-11ea-a78c-180373af3277"
        }
        """
        if not self.is_entity_line(0) and not self.is_entity_line(1):
            self.converter.log_failure("Parallel constraint curves are not lines")
            return None
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
        if self.entity_count == 2 and self.is_entity_point(0) and self.is_entity_point(1):
            return {
                "point_one": self.entities[0]["uuid"],
                "point_two": self.entities[1]["uuid"],
                "type": "VerticalPointsConstraint"
            }
        elif self.entity_count == 1 and self.is_entity_line(0):
            return {
                "line": self.entities[0]["uuid"],
                "type": "VerticalConstraint"
            }
        elif self.entity_count > 1 and self.are_entities_lines():
            # Handle multiple separate constraints
            multi_cst = []
            for ent in self.entities:
                multi_cst.append({
                    "line": ent["uuid"],
                    "type": "VerticalConstraint"
                })
            return multi_cst
        elif self.entity_count > 1 and self.are_entities_points():
            # Handle multiple separate point constraints
            multi_cst = []
            for index in range(self.entity_count - 1):
                multi_cst.append({
                    "point_one": self.entities[index]["uuid"],
                    "point_two": self.entities[index + 1]["uuid"],
                    "type": "VerticalPointsConstraint"
                })
            return multi_cst
        else:
            self.converter.log_failure("Unknown vertical constraint entities")
            return None
   
    def make_tangent_constraint_dict(self):
        """
        {
            "curve_one": "35528476-e0c6-11ea-95eb-c85b76a75ed8",
            "curve_two": "3551c13b-e0c6-11ea-bfd3-c85b76a75ed8",
            "type": "TangentConstraint"
        }
        """
        if not self.is_entity_curve(0) and not self.is_entity_curve(1):
            self.converter.log_failure("Tangent constraint entities are not curves")
            return None
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
        if not self.is_entity_line(0) and not self.is_entity_line(1):
            self.converter.log_failure("Perpendicular constraint entities not lines")
            return None
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
            if not self.is_entity_point(0) and not self.is_entity_curve(1):
                self.converter.log_failure("Midpoint constraint entities not a point and curve")
                return None
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
            self.converter.log_failure("Unknown midpoint constraint entities")
            return None            

    def make_equal_constraint_dict(self):
        """
        {
            "curve_one": "3551e838-e0c6-11ea-a5b3-c85b76a75ed8",
            "curve_two": "3551c13b-e0c6-11ea-bfd3-c85b76a75ed8",
            "type": "EqualConstraint"
        }
        """
        if not self.is_entity_curve(0) and not self.is_entity_curve(1):
            self.converter.log_failure("Equal constraint entities are not curves")
            return None
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
        else:
            self.converter.log_failure("Unknown equal constraint entities")
            return None            


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
            if "parent" in self.entities[0]:
                curve_one = self.entities[0]["parent"]
            else:
                self.converter.log_failure(f"concentricConstraint has unsupported entity {self.entities[0]['type']}")
                return None 
        if self.is_entity_point(1):
            if "parent" in self.entities[1]:
                curve_two = self.entities[1]["parent"]
            else:
                self.converter.log_failure(f"concentricConstraint has unsupported entity {self.entities[1]['type']}")
                return None 

        return {
            "curve_one": curve_one,
            "curve_two": curve_two,
            "type": "ConcentricConstraint"
        }
    
    def make_symmetry_constraint_dict(self):
        """
        {
            "entity_one": "cd825d62-b830-11ea-9e77-180373af3277",
            "entity_two": "cdcf11a4-b830-11ea-bfa8-180373af3277",
            "symmetry_line": "b93bf2dc-b830-11ea-a8ed-180373af3277",
            "type": "SymmetryConstraint"
        },
        """
        symmetry_line = self.entities[0]["uuid"]
        # Loop to handle multiple entity equal constraints
        # e.g.
        # 
        # "mirrorConstraint": {
        #     "mirroredPairs": [
        #         {
        #             "first": 10,
        #             "second": 13
        #         },
        #         {
        #             "first": 11,
        #             "second": 14
        #         }
        #     ]
        # }
        multi_cst = []
        for index in range(1, self.entity_count - 1, 2):
            multi_cst.append({
                "entity_one": self.entities[index]["uuid"],
                "entity_two": self.entities[index + 1]["uuid"],
                "symmetry_line": symmetry_line,
                "type": "SymmetryConstraint"
            })
        if len(multi_cst) == 1:
            return multi_cst[0]
        return multi_cst

    def apply_fix_constraint(self):
        """
        Apply a fix constraint, i.e. fix, to existing curves
        """
        for index, entity in enumerate(self.entities):
            entity_uuid = entity["uuid"]
            # In Fusion, only curves are fixed
            if not self.is_entity_point(index):
                # Set the curve to be fixed in place
                self.curves[entity_uuid]["fixed"] = True
        # Special case flag to indicate we fixed a curve
        # rather than a constraint failure
        return "Fix"