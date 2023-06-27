import uuid

class FusionGalleryConstraint:
    def __init__(self, constraint, points, point_map, curves, entity_map):
        self.points = points
        self.point_map = point_map
        self.curves = curves
        self.entity_map = entity_map
        # UUID of the constraint
        self.uuid = str(uuid.uuid1())
        # Type of constraint in deepmind terms
        self.type = list(constraint.keys())[0]
        # The entities referenced in the deepmind data
        self.entities = []
        for cst_ent_index in constraint[self.type]["entities"]:
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

    def make_coincident_constraint_cases(self):
        assert len(self.entities) == 2
        entity1_type = self.type_for_entity(0)
        entity2_type = self.type_for_entity(1)
        if entity1_type is None or entity2_type is None:
            return None
        if entity1_type == "SketchLine" and entity2_type == "SketchLine":
            return self.make_collinear_constraint_dict()
        if entity1_type == "SketchPoint" or entity2_type == "SketchPoint":
            # Handle the case where two end points are constrained together
            # and we have already deduplicated them
            # So what we want to do is check if the the coincident constraint references
            # the same two points, and skip it is if does            
            if self.entities[0]["uuid"] == self.entities[1]["uuid"]:
                return None
            return self.make_coincident_constraint_dict()
        if entity1_type == "SketchArc" or entity2_type == "SketchArc":
            print("Warning - SketchArc, SketchArc coincident constraint not supported")
            return None
        if entity1_type == "SketchCircle" or entity2_type == "SketchCircle":
            print("Warning - SketchCircle, SketchCircle coincident constraint not supported")
            return None
        assert False, "Unknown case"

    def make_coincident_constraint_dict(self, cst):
        """
        {
            "type": "CoincidentConstraint",
            "entity": "301ed434-b47f-11ea-8e78-180373af3277",
            "point": "30278562-b47f-11ea-be85-180373af3277"
        }
        """
        point, entity = self.find_point_and_entity(cst)
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
            "entity": entity,
            "point": point
        }
        