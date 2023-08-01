import unittest

from PIL import Image

from geometry.visualization import render_sketch_opencv, render_sketch_pil
from tests.test_entities import batch_test_entities


class TestRenderSketchOpenCV(unittest.TestCase):
    def test_render_sketch_opencv(self):
        for entities in batch_test_entities[1:2]:
            np_image = render_sketch_opencv(entities, size=256, quantize_bins=64)
            pil_image = np_image[:, :, ::-1]  # BGR to RGB
            Image.fromarray(pil_image, mode='RGB').show()

    def test_render_sketch_pil(self):
        for entities in batch_test_entities:
            pil_image = render_sketch_pil(entities, figure_size_pixels=512, pad_in_pixels=10, linewidth=2)
            pil_image.show()
