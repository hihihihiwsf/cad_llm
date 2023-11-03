"""

FusionGalleryDimension represents a sketch dimension in the Fusion 360 Gallery format

"""

import math
from preprocess.fusiongallery_geometry.base_constraint import FusionGalleryBaseConstraint


class FusionGalleryDimension(FusionGalleryBaseConstraint):

    # Supported Fusion 360 constraint types
    types = {
        "SketchLinearDimension",
        "SketchDiameterDimension",
        "SketchRadialDimension",
        "SketchAngularDimension",
        "SketchOffsetDimension"
    }     
   
    @staticmethod
    def is_dimension(dm_cst):
        """Determine if an OnShape constraint is a Fusion 360 dimension"""
        cst_type = list(dm_cst.keys())[0]
        dimensions = {
            "distanceConstraint",
            "lengthConstraint",
            "diameterConstraint",
            "radiusConstraint",
            "angleConstraint"
        }
        return cst_type in dimensions

    def to_dict(self):
        """Make a Fusion 360 Gallery format dict for the dimension"""
        dimension_dict = None 
        if self.type == "distanceConstraint":
            dimension_dict = self.make_distance_dimension_dict()
        elif self.type == "lengthConstraint":
            dimension_dict = self.make_length_dimension_dict()
        elif self.type == "diameterConstraint":
            dimension_dict = self.make_diameter_dimension_dict()
        elif self.type == "radiusConstraint":
            dimension_dict = self.make_radius_dimension_dict()
        elif self.type == "angleConstraint":
            dimension_dict = self.make_angle_dimension_dict()
        return dimension_dict
    
    def make_common_dimension_dict(self, value_key="length"):
        """
        {
            "parameter": {
                "type": "ModelParameter",
                "value": 3.0,
                "name": "d3",
                "role": "Linear Dimension-2"
            },
            "is_driving": true
        }
        Value param(kObjectType);
        param.AddMember("type", "ModelParameter", allocator);
        Value param_value(kNumberType);
        param_value.SetDouble(dim->value());
        param.AddMember("value", param_value, allocator);
        dimension_value.AddMember("parameter", param, allocator);
        dimension_value.AddMember("is_driving", true, allocator);
        """
        # Default to zero if not present
        value = self.constraint[self.type].get(value_key, 0)
        dimesion_dict = {
            "parameter": self.make_model_parameter_dict(value),
            "is_driving": True
        }
        return dimesion_dict

    def make_model_parameter_dict(self, value):
        """
        {
            "type": "ModelParameter",
            "value": 3.0,
            "name": "d3",
            "role": "Linear Dimension-2"
        }
        SketchGym only works with the value
        """
        return {
            "type": "ModelParameter",
            "value": value            
        }
    
    def make_orientation_enum(self, direction):
        """
        MINIMUM -> AlignedDimensionOrientation
        HORIZONTAL -> HorizontalDimensionOrientation
        VERTICAL -> VerticalDimensionOrientation
        """
        if direction == "MINIMUM":
            return "AlignedDimensionOrientation"
        elif direction == "HORIZONTAL":
            return "HorizontalDimensionOrientation"
        elif direction == "VERTICAL":
            return "VerticalDimensionOrientation"
        else:
            return None
    
    def calculate_orientation(self, start_point, end_point):
        """Calculate the orientation for a dimension using the start/end points"""
        # We calculate the orientation here as one of:
        # - AlignedDimensionOrientation
        # - HorizontalDimensionOrientation
        # - VerticalDimensionOrientation
        is_horizontal = math.isclose(start_point["y"], end_point["y"])
        is_vertical = math.isclose(start_point["x"], end_point["x"])
        if is_vertical and is_horizontal:
            self.converter.log_failure("Orientation for SketchLinearDimension has coincident start/end points")
            return None
        elif is_vertical:
            return "VerticalDimensionOrientation"
        elif is_horizontal:
            return "HorizontalDimensionOrientation"
        else:
            return "AlignedDimensionOrientation"

    def make_distance_dimension_dict(self):
        """
        An OnShape distance dimension can be one of several types of dimensions
        in Fusion 360, depending on the entities used:

        SketchPoint and SketchPoint - SketchLinearDimension
        SketchLine and SketchLine - SketchOffsetDimension between the two lines
        Point3D and SketchLine - SketchOffsetDimension between the point and line
        SketchCircle and SketchCircle - SketchLinearDimension between the center points
        SketchArc and SketchArc - SketchLinearDimension between the center points
        SketchCircle and SketchLine - CAUSES FUSION ADDIN TO THROW EXCEPTION
        SketchArc and SketchLine - CAUSES FUSION ADDIN TO THROW EXCEPTION
        Point3D and SketchCircle - SketchLinearDimension from the center point
        SketchArc and SketchCircle - SketchLinearDimension between the center points
        Point3D and SketchArc - SketchLinearDimension from the center point

        """        
        # Handle different distance dimension cases
        both_points = self.is_entity_point(0) and self.is_entity_point(1)
        one_point = self.is_entity_point(0) or self.is_entity_point(1)
        one_line = self.is_entity_line(0) or self.is_entity_line(1)
        both_lines = self.is_entity_line(0) and self.is_entity_line(1)
        # Point-Point
        if both_points:
            # TODO: Support circle and arc types
            dimension_dict = self.make_linear_dimension_dict()
        # Point-Line or Line-Line
        elif (one_point and one_line) or both_lines:
            dimension_dict = self.make_offset_dimension_dict(both_lines)
        else:
            dimension_dict = None

        if dimension_dict is None:
            entity_types = sorted([self.entities[0]["type"], self.entities[1]["type"]])
            self.converter.log_failure(f"distanceDimension has unsupported entities {entity_types[0]} and {entity_types[1]}")
            return None

        return dimension_dict
    
    def make_linear_dimension_dict(self):
        """ 
        {
            "parameter": {
                "type": "ModelParameter",
                "value": 20.0,
                "name": "d1",
                "role": "Linear Dimension-2"
            },
            "text_position": {
                "type": "Point3D",
                "x": 5.857106888623424,
                "y": -1.205054404891147,
                "z": 0.0
            },
            "is_driving": true,
            "entity_one": "375ecf3a-e0c6-11ea-b9f4-c85b76a75ed8",
            "entity_two": "375ef648-e0c6-11ea-bbfb-c85b76a75ed8",
            "orientation": "HorizontalDimensionOrientation",
            "type": "SketchLinearDimension"
        }
        """
        dimension_dict = self.make_common_dimension_dict()
        dimension_dict["entity_one"] = self.entities[0]["uuid"] # first
        dimension_dict["entity_two"] = self.entities[1]["uuid"] # second

        # Assume the default is HORIZONTAL, i.e. the 0 enum value here:
        # https://github.com/deepmind/deepmind-research/blob/master/cadl/constraints.proto#L69C5-L69C15
        direction = self.constraint[self.type].get("direction", "HORIZONTAL")
        orientation = self.make_orientation_enum(direction)
        if orientation is None:
            self.converter.log_failure("Unknown direction for distance dimension")
            return None
        dimension_dict["orientation"] = orientation
        dimension_dict["type"] = "SketchLinearDimension"
        return dimension_dict

    def make_offset_dimension_dict(self, both_lines):
        """
        {
            "parameter": {
                "type": "ModelParameter",
                "value": 4.73202,
                "name": "d28_2_1",
                "role": "Linear Dimension-9"
            },
            "text_position": {
                "type": "Point3D",
                "x": 5.726415849460624,
                "y": 2.574062149476403,
                "z": 0.0
            },
            "is_driving": true,
            "line": "7420ba2e-e2b8-11ea-b3a4-54bf646e7e1f",
            "entity_two": "7424611e-e2b8-11ea-b9fc-54bf646e7e1f",
            "type": "SketchOffsetDimension"
        }
        """
        dimension_dict = self.make_common_dimension_dict()
        # Line-Line
        if both_lines:
            dimension_dict["line"] = self.entities[0]["uuid"] # first
            dimension_dict["entity_two"] = self.entities[1]["uuid"] # second
        else:
            # Point-Line
            if self.is_entity_line(0):
                dimension_dict["line"] = self.entities[0]["uuid"]
                dimension_dict["entity_two"] = self.entities[1]["uuid"]
            elif self.is_entity_line(1):
                dimension_dict["line"] = self.entities[1]["uuid"]
                dimension_dict["entity_two"] = self.entities[0]["uuid"]
            else:
                return None
        dimension_dict["type"] = "SketchOffsetDimension"
        return dimension_dict

    def make_length_dimension_dict(self):
        """
        {
            "parameter": {
                "type": "ModelParameter",
                "value": 3.0,
                "name": "d3",
                "role": "Linear Dimension-2"
            },
            "text_position": {
                "type": "Point3D",
                "x": -3.3669161327280506,
                "y": 3.0283618095736404,
                "z": 0.0
            },
            "is_driving": true,
            "entity_one": "355320ca-e0c6-11ea-b4e5-c85b76a75ed8",
            "entity_two": "355320cb-e0c6-11ea-9076-c85b76a75ed8",
            "orientation": "HorizontalDimensionOrientation",
            "type": "SketchLinearDimension"
        }

        """
        if not self.is_entity_line(0):
            self.converter.log_failure(f"lengthDimension has unsupported entity {self.entities[0]['type']}")
            return None
        # We need to find the end points
        start_point = self.entities[0]["start_point"]
        end_point = self.entities[0]["end_point"]
        dimension_dict = self.make_common_dimension_dict()
        dimension_dict["entity_one"] = start_point
        dimension_dict["entity_two"] = end_point
        # Direction is not provided in the deep mind data so we calculate it
        orientation = self.calculate_orientation(self.points[start_point], self.points[end_point])
        if orientation is None:
            return None
        dimension_dict["orientation"] = orientation
        dimension_dict["type"] = "SketchLinearDimension"
        return dimension_dict
    
    def make_diameter_dimension_dict(self):
        """
        {
            "parameter": {
                "type": "ModelParameter",
                "value": 2.236068,
                "name": "d1",
                "role": "Diameter Dimension-2"
            },
            "text_position": {
                "type": "Point3D",
                "x": 12.081785215759233,
                "y": 1.9717289343233286,
                "z": 0.0
            },
            "is_driving": true,
            "curve": "35528477-e0c6-11ea-9b62-c85b76a75ed8",
            "type": "SketchDiameterDimension"
        }
        """
        if not self.is_entity_arc_or_circle(0):
            self.converter.log_failure("Diameter dimension entity is not an arc or circle")
            return None
        diameter_dimension_dict = self.make_common_dimension_dict()
        diameter_dimension_dict["type"] = "SketchDiameterDimension"
        diameter_dimension_dict["curve"] = self.entities[0]["uuid"]
        return diameter_dimension_dict

    def make_radius_dimension_dict(self):
        """
        {
            "parameter": {
                "type": "ModelParameter",
                "value": 2.5,
                "name": "d2",
                "role": "Radial Dimension-2"
            },
            "text_position": {
                "type": "Point3D",
                "x": 14.4136646645875,
                "y": 7.327764543350762,
                "z": 0.0
            },
            "is_driving": true,
            "curve": "35528476-e0c6-11ea-95eb-c85b76a75ed8",
            "type": "SketchRadialDimension"
        }
        """
        if not self.is_entity_arc_or_circle(0):
            self.converter.log_failure("Radius dimension entity is not an arc or circle")
            return None            
        radius_dimension_dict = self.make_common_dimension_dict()
        radius_dimension_dict["type"] = "SketchRadialDimension"
        radius_dimension_dict["curve"] = self.entities[0]["uuid"]
        return radius_dimension_dict
    
    def make_angle_dimension_dict(self):
        """ 
        {
            "parameter": {
                "type": "ModelParameter",
                "value": 0.4960225734167885
            },
            "is_driving": true,
            "type": "SketchAngularDimension",
            "line_one": "25680a1c-b723-11ea-a446-180373af3277",
            "line_two": "2566d1c8-b723-11ea-8520-180373af3277"
        }
        """
        if not self.is_entity_line(0) and not self.is_entity_line(1):
            self.converter.log_failure("Angle dimension entities are not lines")
            return None
        angular_dimension_dict = self.make_common_dimension_dict("angle")
        angular_dimension_dict["type"] = "SketchAngularDimension"
        angular_dimension_dict["line_one"] = self.entities[0]["uuid"]
        angular_dimension_dict["line_two"] = self.entities[1]["uuid"]
        return angular_dimension_dict