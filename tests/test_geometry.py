import unittest
from binvid.geometry import GridSpec, compute_grid, fit_cols_to_width

def check_aspect_ratio_tolerance(src_w: int, src_h: int, out_w: int, out_h: int, max_tolerance: float=0.02) -> None:
    src_ar = src_w / src_h
    out_ar = out_w / out_h
    relative_diff = abs(out_ar - src_ar) / src_ar
    assert relative_diff <= max_tolerance, f'Aspect ratio discrepancy {relative_diff:.4f} ({relative_diff * 100:.2f}%) exceeds tolerance {max_tolerance * 100:.1f}%. src={src_w}x{src_h} (AR={src_ar:.4f}), out={out_w}x{out_h} (AR={out_ar:.4f})'

class TestGeometry(unittest.TestCase):

    def test_16_9_input(self):
        src_w, src_h = (1920, 1080)
        for cols in [100, 150, 200, 240]:
            spec = compute_grid(src_w, src_h, cols=cols, cell_w=8, cell_h=14)
            assert isinstance(spec, GridSpec)
            assert spec.out_w % 2 == 0, f'out_w={spec.out_w} must be even'
            assert spec.out_h % 2 == 0, f'out_h={spec.out_h} must be even'
            check_aspect_ratio_tolerance(src_w, src_h, spec.out_w, spec.out_h, max_tolerance=0.02)

    def test_9_16_vertical_input(self):
        src_w, src_h = (1080, 1920)
        for cols in [80, 100, 120, 135]:
            spec = compute_grid(src_w, src_h, cols=cols, cell_w=8, cell_h=14)
            assert isinstance(spec, GridSpec)
            assert spec.out_w % 2 == 0, f'out_w={spec.out_w} must be even'
            assert spec.out_h % 2 == 0, f'out_h={spec.out_h} must be even'
            check_aspect_ratio_tolerance(src_w, src_h, spec.out_w, spec.out_h, max_tolerance=0.02)

    def test_square_input(self):
        src_w, src_h = (1000, 1000)
        for cols in [80, 100, 150, 200]:
            spec = compute_grid(src_w, src_h, cols=cols, cell_w=8, cell_h=14)
            assert isinstance(spec, GridSpec)
            assert spec.out_w % 2 == 0, f'out_w={spec.out_w} must be even'
            assert spec.out_h % 2 == 0, f'out_h={spec.out_h} must be even'
            check_aspect_ratio_tolerance(src_w, src_h, spec.out_w, spec.out_h, max_tolerance=0.02)

    def test_even_dimensions_with_odd_cells_and_odd_cols(self):
        odd_test_cases = [(1920, 1080, 101, 7, 14), (1920, 1080, 101, 8, 14), (1080, 1920, 99, 7, 14), (1000, 1000, 125, 7, 14)]
        for src_w, src_h, cols, cell_w, cell_h in odd_test_cases:
            spec = compute_grid(src_w, src_h, cols, cell_w, cell_h)
            assert spec.out_w % 2 == 0, f'out_w={spec.out_w} is odd!'
            assert spec.out_h % 2 == 0, f'out_h={spec.out_h} is odd!'

    def test_fit_cols_to_width(self):
        cols_1920 = fit_cols_to_width(src_w=1920, src_h=1080, target_out_w=1920, cell_w=8, cell_h=14)
        spec_1920 = compute_grid(1920, 1080, cols_1920, cell_w=8, cell_h=14)
        assert spec_1920.out_w == 1920
        assert spec_1920.out_h % 2 == 0
        check_aspect_ratio_tolerance(1920, 1080, spec_1920.out_w, spec_1920.out_h)
        cols_1280 = fit_cols_to_width(src_w=1920, src_h=1080, target_out_w=1280, cell_w=8, cell_h=14)
        spec_1280 = compute_grid(1920, 1080, cols_1280, cell_w=8, cell_h=14)
        assert spec_1280.out_w == 1280
        assert spec_1280.out_h % 2 == 0
        cols_vert = fit_cols_to_width(src_w=1080, src_h=1920, target_out_w=1080, cell_w=8, cell_h=14)
        spec_vert = compute_grid(1080, 1920, cols_vert, cell_w=8, cell_h=14)
        assert spec_vert.out_w == 1080
        assert spec_vert.out_h % 2 == 0

    def test_invalid_parameters_raise_value_error(self):
        with self.assertRaises(ValueError):
            compute_grid(0, 1080, 100, 8, 14)
        with self.assertRaises(ValueError):
            compute_grid(1920, -10, 100, 8, 14)
        with self.assertRaises(ValueError):
            compute_grid(1920, 1080, 0, 8, 14)
        with self.assertRaises(ValueError):
            compute_grid(1920, 1080, 100, -8, 14)
        with self.assertRaises(ValueError):
            fit_cols_to_width(1920, 1080, target_out_w=0, cell_w=8, cell_h=14)

def test_pytest_16_9():
    spec = compute_grid(1920, 1080, 200, 8, 14)
    assert spec.out_w % 2 == 0 and spec.out_h % 2 == 0
    check_aspect_ratio_tolerance(1920, 1080, spec.out_w, spec.out_h)

def test_pytest_9_16():
    spec = compute_grid(1080, 1920, 120, 8, 14)
    assert spec.out_w % 2 == 0 and spec.out_h % 2 == 0
    check_aspect_ratio_tolerance(1080, 1920, spec.out_w, spec.out_h)

def test_pytest_square():
    spec = compute_grid(1000, 1000, 150, 8, 14)
    assert spec.out_w % 2 == 0 and spec.out_h % 2 == 0
    check_aspect_ratio_tolerance(1000, 1000, spec.out_w, spec.out_h)
if __name__ == '__main__':
    unittest.main()
