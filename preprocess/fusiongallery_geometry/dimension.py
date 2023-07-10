"""

FusionGalleryDimension represents a sketch dimension in the Fusion 360 Gallery format

"""

from preprocess.fusiongallery_geometry.base_constraint import FusionGalleryBaseConstraint


class FusionGalleryDimension(FusionGalleryBaseConstraint):

    # Supported Fusion 360 constraint types
    types = {
        "SketchLinearDimension",
        "SketchDiameterDimension",
        "SketchRadialDimension",
        "SketchAngularDimension"
    }     

    def __init__(self, constraint, points, curves, entity_map):
        """
        Intialize a FusionGalleryDimension

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
            assert False, "Unknown direction"

    def make_distance_dimension_dict(self):
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
        if not self.is_entity_point_or_line(0) and not self.is_entity_point_or_line(1):
            print("Warning! - SketchGym only supports point-point, point-line or line-line distance constraints")
            return None
        dimension_dict = self.make_common_dimension_dict()
        if dimension_dict is None:
            return None
        dimension_dict["entity_one"] = self.entities[0]["uuid"] # first
        dimension_dict["entity_two"] = self.entities[1]["uuid"] # second
        # Assume the default is HORIZONTAL, i.e. the 0 enum value here:
        # https://github.com/deepmind/deepmind-research/blob/master/cadl/constraints.proto#L69C5-L69C15
        direction = self.constraint[self.type].get("direction", "HORIZONTAL")
        dimension_dict["orientation"]= self.make_orientation_enum(direction)
        dimension_dict["type"] = "SketchLinearDimension"
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
            print("Warning - Length dimension not used on line")
            return None
        # We need to find the end points
        start_point = self.entities[0]["start_point"]
        end_point = self.entities[0]["end_point"]
        dimension_dict = self.make_common_dimension_dict()
        if dimension_dict is None:
            return None
        dimension_dict["entity_one"] = start_point
        dimension_dict["entity_two"] = end_point
        # Direction is not provided in the deep mind data
        # dimension_dict["orientation"]= self.make_orientation_enum(self.constraint[self.type]["direction"])
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
        assert self.is_entity_arc_or_circle(0)
        diameter_dimension_dict = self.make_common_dimension_dict()
        if diameter_dimension_dict is None:
            return None
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
        assert self.is_entity_arc_or_circle(0)
        radius_dimension_dict = self.make_common_dimension_dict()
        if radius_dimension_dict is None:
            return None
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
        assert self.is_entity_line(0)
        assert self.is_entity_line(1)
        angular_dimension_dict = self.make_common_dimension_dict("angle")
        if angular_dimension_dict is None:
            return None
        angular_dimension_dict["type"] = "SketchAngularDimension"
        angular_dimension_dict["line_one"] = self.entities[0]["uuid"]
        angular_dimension_dict["line_two"] = self.entities[1]["uuid"]
        return angular_dimension_dict