import uuid

class DeepmindBase:
    def __init__(self, dm_ent):
        self.is_construction = dm_ent.get("isConstruction", False)
        self.exception = None
        self.dm_ent = dm_ent
        self.color = "black"
        self.uuid = str(uuid.uuid1())
