import unittest

from PIL import Image

from geometry.visualization import render_sketch_opencv
from tests.test_entities import batch_test_entities


class TestRenderSketchOpenCV(unittest.TestCase):
    def test_get_rendered_sketch_dataset(self):

        for entities in batch_test_entities:
            np_image = render_sketch_opencv(entities, size=256, quantize_bins=64)
            pil_image = np_image[:, :, ::-1]  # BGR to RGB
            Image.fromarray(pil_image, mode='RGB').show()
