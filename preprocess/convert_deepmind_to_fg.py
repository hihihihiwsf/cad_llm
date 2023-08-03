"""

Convert Deepmind sketch dataset sketches
into the Fusion 360 Gallery dataset format

The Deepmind sketch dataset comes in a protobuf format.
Before using this script either convert from protobuf to json using:
    preprocess/convertprotobuf.py
or get the converted json files from someone who has already
suffered through this process and taken one for the team.

"""

import argparse
import json
import uuid
from pathlib import Path
from tqdm import tqdm

from preprocess.deepmind_geometry import *
from preprocess.fusiongallery_geometry import *
from preprocess.preprocess_utils import get_files, get_output_dir
from tests.test_fusiongallery_sketch import TestFusionGallerySketch
from preprocess.converter_base import ConverterBase


class DeepmindToFusionGalleryConverter(ConverterBase):
    """
    Class to handle conversion of Deepmind sketches
    into the Fusion 360 Gallery dataset format
    """
    def __init__(self, input_files, output_path, limit):
        super().__init__()
        self.input_files = input_files
        self.output_path = output_path
        self.limit = limit

    def convert(self):
        """Convert all of the input files"""
        # The data comes in 96 different data files
        # that we loop over here
        for i, input_file in enumerate(self.input_files):
            print(f"Converting {i}/{len(self.input_files)} data files")
            with open(input_file) as f:
                dm_data = json.load(f)
            self.convert_data(dm_data, input_file)
            if args.limit is not None and self.converted_count >= args.limit:
                break    
        print(f"Converted {self.converted_count}/{self.count} sketches!")

    def convert_data(self, dm_data, input_file):
        """Convert all the sketches in a single data file and save them as multiple json files"""
        for index, dm_sketch in enumerate(tqdm(dm_data)):
            fg_sketch = self.convert_sketch(dm_sketch)
            if fg_sketch is not None:
                # Save the file with the name of the original deepmind data file
                # and the index into that file of the sketch
                json_file = self.output_path / f"{input_file.stem}_{index:06d}.json"
                with open(json_file, "w") as f:
                    json.dump(fg_sketch, f, indent=4)
                if json_file.exists():
                    self.converted_count += 1
            self.count += 1
            if self.limit is not None and self.converted_count >= self.limit:
                break

    def convert_sketch(self, dm_sketch):
        """Convert a single sketch in the deepmind sketch format into a Fusion Gallery sketch dict"""
        dm_entities = dm_sketch["entitySequence"]["entities"]
        dm_constraints = dm_sketch["constraintSequence"]["constraints"]
        
        try:
            # First check that the sketch is good to convert
            # This will raise an exception if there are issues
            self.is_sketch_good(dm_entities)
            points, point_map = self.create_sketch_points(dm_entities)
            curves, constraint_entity_map = self.create_sketch_curves(dm_entities, point_map)
            constraints, dimensions = self.create_constraints_and_dimensions(dm_constraints, points, curves, constraint_entity_map)
            fusion_gallery_sketch = {
                "name": f"Sketch{self.count}", # Save the sketch with the overall index in the dataset
                "type": "Sketch",
                "points": points,
                "curves": curves,
                "constraints": constraints,
                "dimensions": dimensions
            }
            #  Test the converted sketch for self-consistency
            sketch_test = TestFusionGallerySketch()
            sketch_test.run_sketch_test(fusion_gallery_sketch)

        except Exception as ex:
            self.log_failure(ex)
            return None
        return fusion_gallery_sketch

    def is_sketch_good(self, dm_entities):
        """Check that a sketch is valid and supported conversion"""
        curve_count = 0
        for dm_ent in dm_entities:
            entity_type = list(dm_ent.keys())[0]
            if entity_type == "pointEntity":
                # Skip
                pass
            elif entity_type == "lineEntity":
                curve_count += 1
            elif entity_type == "circleArcEntity":
                curve_count += 1
            elif entity_type == "interpolatedSplineEntity":
                raise Exception("interpolatedSplineEntity not supported")
            else:
                raise Exception(f"Unexpected entity type not supported: {entity_type}")                
        if curve_count == 0:
            raise Exception("Sketch doesn't have any curves")

    @staticmethod
    def create_sketch_points(dm_entities):
        """Create the sketch points data structure"""
        # point_map.map containts a dict of unique points
        # key: string of the form point.x_point.y
        # value: FusionGalleryPoint object
        point_map = FusionGalleryPointMap(dm_entities)
        # Sketch points data structure in the FG format
        point_data = {}
        for key, fg_point in point_map.map.items():
            point_data[fg_point.uuid] = fg_point.to_dict()
        return point_data, point_map  

    @staticmethod
    def create_sketch_curves(dm_entities, point_map):
        """Create the sketch curves data structure"""
        # Curve dictionary as stored in FG json format
        curve_data = {}
        # Mapping from the constraint entities index array to uuids in either the points or curves dict
        constraint_entity_map = {}
        constraint_entity_index = 0
        for dm_ent in dm_entities:
            assert len(dm_ent) == 1, "Expected on entry in the dict"
            entity_type = list(dm_ent.keys())[0]
            entity_data = dm_ent[entity_type]
            add_curve = False

            if entity_type == "pointEntity":
                dm_obj = DeepmindPoint(entity_data)
                # Find the point in the point map
                fg_obj = FusionGalleryPoint.from_xy_map(
                    dm_obj.x,
                    dm_obj.y,
                    point_map.map
                )

            elif entity_type == "lineEntity":
                dm_obj = DeepmindLine(entity_data)
                fg_obj = FusionGalleryLine(dm_obj, point_map.map)
                add_curve = True

            elif entity_type == "circleArcEntity":
                if "arcParams" in entity_data:
                    dm_obj = DeepmindArc(entity_data)
                    fg_obj = FusionGalleryArc(dm_obj, point_map.map)
                    add_curve = True
                elif "circleParams" in entity_data:
                    dm_obj = DeepmindCircle(entity_data)
                    fg_obj = FusionGalleryCircle(dm_obj, point_map.map)
                    add_curve = True

            # Generate the FG dict that will get saved to json for the entity
            fg_dict = fg_obj.to_dict()
            # Register the entity so we can map between the entity indices given in the deepmind
            # data and the FG curve and point dicts
            constraint_entity_map, constraint_entity_index = DeepmindToFusionGalleryConverter.update_constraint_entity_map(
                constraint_entity_map,
                fg_obj,
                fg_dict,
                constraint_entity_index
            )
            # Add the entity to the FG curve dict if it is a line, arc, circle
            if add_curve:
                curve_data[fg_obj.uuid] = fg_dict

        return curve_data, constraint_entity_map

    @staticmethod
    def update_constraint_entity_map(entity_map, fg_obj, fg_dict, index):
        """
        Update the given entity in the constraint entity map

        The key in the entity_map is an index counted based on a specific layout:

        entity0.part0, entity0.part1, ..., entity0.partN, entity1.part0, entity1.part1, ...

        The following entity types and parts are used in the layout

        Entity type                | Parts
        ---------------------------+-------------------------------------------------------------
        point_entity               | entity
        interpolated_spline_entity | entity, interp_point0, ..., interp_pointM, start_derivative,
                                |     end_derivative, start_point, end_point
        circle_arc_entity          | entity, center, [start_point, end_point] (if not closed)
        line_entity                | entity, start_point, end_point

        See: https://github.com/deepmind/deepmind-research/issues/377#issuecomment-1218204488
        
        """
        updated_index = index

        if fg_dict["type"] == "Point3D":
            # entity
            entity_map[updated_index] = {
                "type": "point",
                "uuid": fg_obj.uuid
            }
            updated_index += 1
        
        elif fg_dict["type"] == "SketchCircle":
            # entity
            entity_map[updated_index] = {
                "type": "curve",
                "uuid": fg_obj.uuid
            }
            updated_index += 1
            # center
            entity_map[updated_index] = {
                "type": "point",
                "uuid": fg_dict["center_point"],
                "parent": fg_obj.uuid
            }
            updated_index += 1

        elif fg_dict["type"] == "SketchArc":
            # entity
            entity_map[updated_index] = {
                "type": "curve",
                "uuid": fg_obj.uuid
            }
            updated_index += 1
            # center
            entity_map[updated_index] = {
                "type": "point",
                "uuid": fg_dict["center_point"],
                "parent": fg_obj.uuid
            }
            updated_index += 1
            # start_point
            entity_map[updated_index] = {
                "type": "point",
                "uuid": fg_dict["start_point"],
                "parent": fg_obj.uuid
            }
            updated_index += 1
            # end_point
            entity_map[updated_index] = {
                "type": "point",
                "uuid": fg_dict["end_point"],
                "parent": fg_obj.uuid
            }
            updated_index += 1
        
        elif fg_dict["type"] == "SketchLine":
            # entity
            # Store references to the start and end points
            entity_map[updated_index] = {
                "type": "curve",
                "uuid": fg_obj.uuid,
                "start_point": fg_dict["start_point"],
                "end_point": fg_dict["end_point"],
            }
            updated_index += 1
            # start_point
            entity_map[updated_index] = {
                "type": "point",
                "uuid": fg_dict["start_point"],
                "parent": fg_obj.uuid
            }
            updated_index += 1
            # end_point
            entity_map[updated_index] = {
                "type": "point",
                "uuid": fg_dict["end_point"],
                "parent": fg_obj.uuid
            }
            updated_index += 1
        
        else:
            print("Warning - unexpected entity type", fg_dict["type"])

        return entity_map, updated_index

    def create_constraints_and_dimensions(self, dm_constraints, points, curves, constraint_entity_map):
        """Create the constraints and dimensions data structure"""
        constraints_data = {}
        dimensions_data = {}
        # Whether all constraints converted sucessfully
        all_convert_success = True
        for constraint in dm_constraints:
            if FusionGalleryDimension.is_dimension(constraint):
                convert_success = self.create_dimension(constraint, points, curves, constraint_entity_map, dimensions_data)
            else:
                convert_success = self.create_constraint(constraint, points, curves, constraint_entity_map, constraints_data)
            if not convert_success:
                all_convert_success = False
        # If all constraints converted, log it
        if all_convert_success:
            self.perfect_sketch_converted_count += 1
        return constraints_data, dimensions_data

    def create_constraint(self, constraint, points, curves, constraint_entity_map, constraints_data):
        """Create a constraint and add it to the provided constraints_data dictionary"""
        constraint = FusionGalleryConstraint(constraint, points, curves, constraint_entity_map, converter=self)
        cst_dict_or_list = constraint.to_dict()
        # Don't count merged points as conversion failures
        if cst_dict_or_list != "Merge":
            self.constraint_count += 1
            if cst_dict_or_list is not None:
                # Ignore multiple constraint counts
                self.constraint_converted_count += 1
                # Single constraint
                if isinstance(cst_dict_or_list, dict):
                    constraints_data[constraint.uuid] = cst_dict_or_list
                # Multiple constraint case
                elif isinstance(cst_dict_or_list, list):
                    for cst_dict in cst_dict_or_list:
                        if cst_dict is not None:
                            cst_uuid = str(uuid.uuid1())
                            constraints_data[cst_uuid] = cst_dict
        return cst_dict_or_list is not None

    def create_dimension(self, dimension, points, curves, constraint_entity_map, dimensions_data):
        """Create a constraint and add it to the provided constraints_data dictionary"""
        self.dimension_count += 1
        dimension = FusionGalleryDimension(dimension, points, curves, constraint_entity_map, converter=self)
        dimension_dict = dimension.to_dict()
        if dimension_dict is not None:
            dimensions_data[dimension.uuid] = dimension_dict
            self.dimension_converted_count += 1
        return dimension_dict is not None

def main(args):
    input_path = Path(args.input)
    if input_path.is_dir():
        input_files = get_files(args.input, pattern="*.json")
        input_files.sort()
    elif input_path.is_file():
        input_files = [input_path]

    output_path = get_output_dir(args.output)
    converter = DeepmindToFusionGalleryConverter(input_files, output_path, args.limit)
    converter.convert()
    converter.print_log_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input file/folder of DeepMind json files")
    parser.add_argument("--output", type=str, help="Output folder to save the Fusion 360 Gallery json data [default: output]")
    parser.add_argument("--limit", type=int, help="Limit the number of files to process")
    args = parser.parse_args()
    main(args)
