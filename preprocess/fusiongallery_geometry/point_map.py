"""

FusionGalleryPointMap that maps between unique keys and FusionGalleryPoints

"""


from preprocess.deepmind_geometry.point import DeepmindPoint
from preprocess.fusiongallery_geometry.point import FusionGalleryPoint


class FusionGalleryPointMap():
    
    def __init__(self, dm_entities):
        """
        Create a FusionGalleryPointMap that maps between unique keys/uuids and FusionGalleryPoints
        These maps can be accessed using self.map and self.uuid_map

        Args
            dm_entities (list): Entities from the DeepMind sketch dataset, 
                            i.e. ["entitySequence"]["entities"]
        """
        # self.map containts a dict of unique points
        #   - key: string of the form point.x_point.y
        #   - value: FusionGalleryPoint object
        self.map = {}

        # self.uuid_map containts a dict of unique points
        #   - key: unique uuid string
        #   - value: FusionGalleryPoint object
        self.uuid_map = {}
        
        point_count = 0
        for dm_ent in dm_entities:
            points = FusionGalleryPointMap.find_all_geom_points(dm_ent)
            for point in points:
                point_count += 1
                # First make a deepmind point then pass that to our fusion gallery point
                dm_point = DeepmindPoint(point)
                fg_point = FusionGalleryPoint(dm_point)
                if not fg_point.key in self.map:
                    self.map[fg_point.key] = fg_point
        # print(f"Created vertex dictionary with {len(self.map)} of {point_count} total vertices")
        for key, value in self.map.items():
            self.uuid_map[value.uuid] = value


    @staticmethod
    def find_all_geom_points(dm_ent):
        """Given an OnShape entity, return all of its points"""
        assert len(dm_ent) == 1, "Expected on entry in the dict"
        entity_type = FusionGalleryPointMap.get_dm_ent_name(dm_ent)
        entity_data = dm_ent[entity_type]

        if entity_type == "pointEntity":
            return [entity_data["point"]]
        if entity_type == "lineEntity":
            return [entity_data["start"], entity_data["end"]]
        if entity_type == "circleArcEntity":
            if "arcParams" in entity_data:
                return [
                    entity_data["center"],
                    entity_data["arcParams"]["start"],
                    entity_data["arcParams"]["end"]
                ]
            elif "circleParams" in entity_data:
                return [entity_data["center"]]            

    @staticmethod
    def get_dm_ent_name(dm_ent):
        return list(dm_ent.keys())[0]