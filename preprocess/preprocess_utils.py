from pathlib import Path
import json


def get_files(input_path, pattern):
    """Get the Sketch Graphs files to process"""
    input_dir = Path(input_path)
    if not input_dir.exists():
        print("Input folder does not exist")
        exit()
    if not input_dir.is_dir():
        print("Input folder is not a directory")
        exit()
    files = [f for f in input_dir.glob(pattern)]
    # e.g. files = [ input_dir / "sg_t16_validation.npy" ]
    if len(files) == 0:
        print("No SketchGraphs files found")
        exit()
    return files


def get_output_dir(output_path):
    """Get the output directory to save the data"""
    current_dir = Path(__file__).resolve().parent
    if output_path is not None:
        output_dir = Path(output_path)
    else:
        output_dir = current_dir / "output"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    return output_dir


def load_filter(path):
    """
        Expected filter file format:
        {
            'train': ['train_07265623.obj', ...],
            'val': ['val_00032698.obj', ...],
            'test': ['test_00289446.obj', ...]
        }
    """
    with open(path) as f:
        split_to_filenames = json.load(f)
    return split_to_filenames


def save_splits(output_dir, split_to_sketches):
    for split_name, sketches in split_to_sketches.items():
        filename = output_dir / f"{split_name}.json"
        with open(filename, "w") as f:
            json.dump(sketches, f)


def sort_points(points):
    if not points:
        return None

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
    return tuple(points)


def point_entity_from_flat_points(flat_points, sort):
    """
    :param flat_points: list of points in the form [x1, y1, x2, y2, ...]
    :param sort: whether to sort the points
    :return: list of points in the form [(x1, y1), (x2, y2), ...]
    """
    if len(flat_points) % 2 != 0:
        flat_points = flat_points[:-1]

    point_entity = [(flat_points[i], flat_points[i + 1]) for i in range(0, len(flat_points), 2)]
    if sort:
        point_entity = sort_points(point_entity)

    return point_entity
