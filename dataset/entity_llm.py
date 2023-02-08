class EntityLLM:
    def __init__(self, points):
        self.points = self._sort_points(points)

    @staticmethod
    def _sort_points(points):
        if len(points) == 2:  # Line
            points = sorted(points)
        elif len(points) == 3:  # Arc
            start, mid, end = points
            if start > end:
                points = [end, mid, start]
        if len(points) == 4:  # Circle
            # top, right, bottom, left = points
            # sort -> left, top, bottom, right
            points = sorted(points)
        return points

    def to_string(self):
        return ",".join(f"{x},{y}" for x, y in self.points) + ";"
