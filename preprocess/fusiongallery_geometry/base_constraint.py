"""

FusionGalleryBaseConstraint represents a base for sketch constraints and dimensions
in the Fusion 360 Gallery format

"""

import uuid


class FusionGalleryBaseConstraint:
    def __init__(self, constraint, points, curves, entity_map, converter=None):
        """
        Intialize a FusionGalleryBaseConstraint

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
            converter (DeepmindToFusionGalleryConverter): Parent converter class
        """
        self.constraint = constraint
        self.points = points
        self.curves = curves
        self.entity_map = entity_map
        self.converter = converter
        # UUID of the constraint
        self.uuid = str(uuid.uuid1())
        # Type of constraint in deepmind terms
        self.type = list(self.constraint.keys())[0]

        # Standardize the list of entity indices
        dm_entities = self.prepare_entity_indices()
        # The entities referenced in the deepmind data
        # stored in order of their index, in FG dict format
        # with the addition of a uuid key
        self.entities, self.entity_count = self.prepare_entity_list(dm_entities)
    
    def to_dict(self):
        """Make a Fusion 360 Gallery format dict for the constraint/dimension"""
        # To be implemented by the child class
        pass

    def prepare_entity_list(self, dm_entities):
        """Prepare a list of entities that matches the indices in the deepmind data"""
        entities = []
        entity_count = 0

        for cst_ent_index in dm_entities:
            # Check that the referenced entity actually exists
            if cst_ent_index not in self.entity_map:
                entities.append(None)
                continue
            # Type is either point or curve
            cst_type = self.entity_map[cst_ent_index]["type"]
            # UUID into either the points or curves dicts
            cst_uuid = self.entity_map[cst_ent_index]["uuid"]
            if cst_type == "point":
                ent = self.points[cst_uuid]
            elif cst_type == "curve":
                ent = self.curves[cst_uuid]
            # New dictionary with the additional uuid value
            ent_data = {
                **ent,
                "uuid": cst_uuid
            }
            # Add the parent reference for lines/arcs/circles
            if "parent" in self.entity_map[cst_ent_index]:
                ent_data["parent"] = self.entity_map[cst_ent_index]["parent"]
            # Add the start and end point references
            if "start_point" in self.entity_map[cst_ent_index]:
                ent_data["start_point"] = self.entity_map[cst_ent_index]["start_point"]
            if "end_point" in self.entity_map[cst_ent_index]:
                ent_data["end_point"] = self.entity_map[cst_ent_index]["end_point"]                
            
            entities.append(ent_data)
            entity_count += 1
        
        return entities, entity_count

    def prepare_entity_indices(self):
        """Standardize the list of entity indices from the deepmind constraint data"""
        entities = []
        # Note we have to default to the 0 index entity in case it is missing
        # due to the protobuf BS

        # CONSTRAINTS
        if self.type == "tangentConstraint" or self.type == "perpendicularConstraint":
            entities = [
                self.constraint[self.type].get("first", 0),
                self.constraint[self.type].get("second", 0)
            ]
        elif self.type == "midpointConstraint":
            entities = [
                self.constraint[self.type].get("midpoint", 0)
            ]
            if "endpoints" in self.constraint[self.type]:
                entities.append(self.constraint[self.type]["endpoints"].get("first", 0))
                entities.append(self.constraint[self.type]["endpoints"].get("second", 0))
            else:
                entities.append(self.constraint[self.type].get("entity", 0))
        
        # DIMENSIONS
        elif self.type == "distanceConstraint" or self.type == "angleConstraint":
            entities = [
                self.constraint[self.type].get("first", 0),
                self.constraint[self.type].get("second", 0)
            ]
        elif self.type == "lengthConstraint" or self.type == "diameterConstraint" or self.type == "radiusConstraint":
            # Default to the 0 index entity if it is missing
            entities = [
                self.constraint[self.type].get("entity", 0)
            ]
        
        else:
            if "entities" in self.constraint[self.type]:
                entities = self.constraint[self.type]["entities"]
        return entities

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

    def are_entities_lines(self):
        for ent in self.entities:
            if ent["type"] != "SketchLine":
                return False
        return True

    def are_entities_points(self):
        for ent in self.entities:
            if ent["type"] != "Point3D":
                return False
        return True

    def is_entity_line(self, index):
        return self.is_entity_type(index, "SketchLine")

    def is_entity_arc(self, index):
        return self.is_entity_type(index, "SketchArc")

    def is_entity_circle(self, index):
        return self.is_entity_type(index, "SketchCircle")
    
    def is_entity_curve(self, index):
        ent_type = self.entities[index]["type"]
        return ent_type == "SketchLine" or ent_type == "SketchArc" or ent_type == "SketchCircle"

    def is_entity_point_or_line(self, index):
        return self.is_entity_line(index) or self.is_entity_point(index)
    
    def is_entity_arc_or_circle(self, index):
        return self.is_entity_arc(index) or self.is_entity_circle(index)