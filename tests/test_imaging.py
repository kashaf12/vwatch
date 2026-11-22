import base64
import io

import numpy as np
import pytest
from PIL import Image

from vwatch import imaging
from vwatch.imaging import Box


class TestBox:
    def test_valid(self):
        assert Box(0, 0, 1, 1) == imaging.FULL

    @pytest.mark.parametrize("args", [(0.5, 0, 0.5, 1), (0.6, 0, 0.5, 1), (-0.1, 0, 1, 1),
                                      (0, 0, 1.1, 1), (0, 0.9, 1, 0.1), (0, 0, 1, 1.5)])
    def test_invalid(self, args):
        with pytest.raises(ValueError, match="invalid normalized box"):
            Box(*args)

    def test_clamp_out_of_range(self):
        assert Box.clamp(-1, -1, 2, 2) == Box(0, 0, 1, 1)

    def test_clamp_reversed(self):
        assert Box.clamp(0.8, 0.9, 0.2, 0.1) == Box(0.2, 0.1, 0.8, 0.9)

    def test_clamp_min_size(self):
        b = Box.clamp(0.5, 0.5, 0.5, 0.5)
        assert b.x1 - b.x0 == pytest.approx(0.02) and b.y1 - b.y0 == pytest.approx(0.02)

    def test_clamp_near_edge_stays_valid(self):
        b = Box.clamp(0.99, 0.99, 1.0, 1.0)
        assert 0 <= b.x0 < b.x1 <= 1

    def test_clamp_degenerate_at_edge(self):
        b = Box.clamp(1.0, 0.0, 1.0, 1.0)
        assert b.x0 < b.x1 <= 1


class TestTokens:
    def test_small(self):
        assert imaging.estimate_image_tokens(100, 100) == 14  # ceil(10000/750)

    def test_no_resize_at_limit(self):
        assert imaging.estimate_image_tokens(1568, 1568) == 3279

    def test_resize_above_limit(self):
        # 3136x1568 -> 1568x784
        assert imaging.estimate_image_tokens(3136, 1568) == 1640

    def test_resize_tall(self):
        assert imaging.estimate_image_tokens(1000, 3136) == imaging.estimate_image_tokens(500, 1568)

    def test_minimum_one(self):
        assert imaging.estimate_image_tokens(1, 1) == 1

    def test_text_tokens(self):
        assert imaging.estimate_text_tokens("") == 1
        assert imaging.estimate_text_tokens("abcdefg") == 3


class TestCropResize:
    def test_crop_shape_and_copy(self):
        f = np.arange(100 * 200 * 3, dtype=np.uint32).astype(np.uint8).reshape(100, 200, 3)
        c = imaging.crop(f, Box(0.25, 0.5, 0.75, 1.0))
        assert c.shape == (50, 100, 3)
        np.testing.assert_array_equal(c, f[50:100, 50:150])
        c[:] = 0
        assert f[50:100, 50:150].any()  # not a view

    def test_crop_tiny_box_at_least_one_pixel(self):
        f = np.zeros((10, 10, 3), dtype=np.uint8)
        c = imaging.crop(f, Box(0.5, 0.5, 0.51, 0.51))
        assert c.shape[0] >= 1 and c.shape[1] >= 1

    def test_resize_width(self):
        f = np.zeros((40, 20, 3), dtype=np.uint8)
        assert imaging.resize_width(f, 10).shape == (20, 10, 3)
        assert imaging.resize_width(f, 20) is f


def _tiles(n, h=10, w=8):
    return [np.full((h, w, 3), 10 * (k + 1), dtype=np.uint8) for k in range(n)]


class TestGrid:
    def test_shape_and_layout(self):
        g = imaging.grid(_tiles(5), 2, gap=4)
        assert g.shape == (3 * 10 + 2 * 4, 2 * 8 + 4, 3)
        assert (g[0:10, 0:8] == 10).all() and (g[0:10, 12:20] == 20).all()
        assert (g[14:24, 0:8] == 30).all()

    def test_padding_cell_is_grey(self):
        g = imaging.grid(_tiles(5), 2, gap=4)
        assert (g[28:38, 12:20] == 90).all()

    def test_cols_clamped_to_count(self):
        assert imaging.grid(_tiles(2), 9).shape == (10, 2 * 8 + 4, 3)

    def test_single(self):
        assert imaging.grid(_tiles(1), 4).shape == (10, 8, 3)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            imaging.grid([], 2)

    def test_mismatched_tile_resized(self):
        tiles = _tiles(2)
        tiles[1] = np.full((20, 16, 3), 200, dtype=np.uint8)
        g = imaging.grid(tiles, 2)
        assert g.shape == (10, 20, 3) and (g[:, 12:20] == 200).all()


class TestStamp:
    def test_new_array_no_mutation(self):
        f = np.full((80, 60, 3), 50, dtype=np.uint8)
        before = f.copy()
        out = imaging.stamp(f, "00:01.00")
        np.testing.assert_array_equal(f, before)
        assert out is not f and out.shape == f.shape
        assert (out[:8, :8] != 50).any()  # label drawn top-left
        assert (out[-10:, -10:] == 50).all()

    def test_gray_input_becomes_rgb(self):
        out = imaging.stamp(np.zeros((40, 40), dtype=np.uint8), "x")
        assert out.shape == (40, 40, 3)


class TestMisc:
    @pytest.mark.parametrize("t,s", [(0, "00:00.00"), (65.5, "01:05.50"), (-3, "00:00.00"),
                                     (3599.99, "59:59.99")])
    def test_fmt_time(self, t, s):
        assert imaging.fmt_time(t) == s

    def test_image_block(self):
        f = np.full((30, 20, 3), 128, dtype=np.uint8)
        blk = imaging.image_block(f)
        assert blk["type"] == "image" and blk["source"]["media_type"] == "image/jpeg"
        img = Image.open(io.BytesIO(base64.b64decode(blk["source"]["data"])))
        assert img.format == "JPEG" and img.size == (20, 30)

    def test_image_block_gray(self):
        blk = imaging.image_block(np.zeros((10, 10), dtype=np.uint8))
        assert blk["source"]["data"]


def test_font_falls_back_to_default(monkeypatch):
    orig = imaging.ImageFont.truetype

    def nope(font=None, *a, **k):
        if isinstance(font, str) and font.endswith(".ttf"):
            raise OSError("no font")
        return orig(font, *a, **k)  # Pillow's built-in default font loads via truetype too

    monkeypatch.setattr(imaging.ImageFont, "truetype", nope)
    out = imaging.stamp(np.zeros((40, 40, 3), dtype=np.uint8), "t")
    assert out.shape == (40, 40, 3)
