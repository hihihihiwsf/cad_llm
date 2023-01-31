class Entity:
    def __init__(self, points):
        self.points = points

    def _sort_points(self):
        if len(self.points) == 2:
            self.points = sorted(self.points)
        elif len(self.points) == 3:
            start, mid, end = self.points
            if start > end:
                self.points = [end, mid, start]
        if len(self.points) == 4:
            # top, right, bottom, left = self.points
            # sort -> left, top, bottom, right
            self.points = sorted(self.points)
        return self.points

    def to_string(self):
        self._sort_points()
        return ",".join(f"{x},{y}" for x, y in self.points) + ";"

    @staticmethod
    def entities_to_string(entities):
        return "".join(ent.to_string() for ent in entities)
