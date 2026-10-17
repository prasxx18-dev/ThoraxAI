"""Image loading, conversion and validation."""

import io

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import ImageValidationError, load_image, prepare_array, prepare_batch, verify_image_file


def _png_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_grayscale_converted_to_rgb():
    img = load_image(_png_bytes(Image.new("L", (100, 120), 128)), filename="x.png")
    assert img.mode == "RGB" and img.size == (100, 120)
    arr = np.asarray(img)
    assert np.all(arr[..., 0] == arr[..., 1]) and np.all(arr[..., 1] == arr[..., 2])


def test_16bit_and_rgba_images():
    arr16 = (np.arange(100 * 100, dtype=np.uint16).reshape(100, 100))
    img = load_image(_png_bytes(Image.fromarray(arr16)), filename="a.png")
    assert img.mode == "RGB"
    rgba = load_image(_png_bytes(Image.new("RGBA", (80, 80), (10, 20, 30, 255))), filename="b.png")
    assert rgba.mode == "RGB"


def test_jpeg_supported():
    img = load_image(_png_bytes(Image.new("RGB", (90, 90), (50, 50, 50)), "JPEG"), filename="c.jpeg")
    assert img.mode == "RGB"


def test_resize_and_value_range():
    arr = prepare_array(Image.new("RGB", (300, 200), (255, 0, 0)), 64)
    assert arr.shape == (64, 64, 3) and arr.dtype == np.float32
    assert arr.max() <= 255 and arr.min() >= 0
    assert prepare_batch(Image.new("RGB", (100, 100)), 32).shape == (1, 32, 32, 3)


@pytest.mark.parametrize(
    "data, name, category",
    [
        (b"garbage bytes", "x.png", "corrupted"),
        (b"", "x.png", "empty"),
        (b"anything", "x.gif", "unsupported_format"),
    ],
)
def test_invalid_uploads_rejected(data, name, category):
    with pytest.raises(ImageValidationError) as exc:
        load_image(data, filename=name)
    assert exc.value.category == category


def test_too_small_and_too_large():
    with pytest.raises(ImageValidationError) as exc:
        load_image(_png_bytes(Image.new("L", (20, 20))), filename="s.png")
    assert exc.value.category == "too_small"
    with pytest.raises(ImageValidationError) as exc:
        load_image(b"0" * 100, filename="big.png", max_bytes=10)
    assert exc.value.category == "too_large"


def test_truncated_png_rejected():
    data = _png_bytes(Image.new("L", (200, 200), 5))
    with pytest.raises(ImageValidationError) as exc:
        load_image(data[: len(data) // 2], filename="t.png")
    assert exc.value.category == "corrupted"


def test_verify_image_file(tmp_path):
    good = tmp_path / "g.png"
    Image.new("L", (64, 64)).save(good)
    bad = tmp_path / "b.png"
    bad.write_bytes(b"\x89PNG broken")
    assert verify_image_file(good) == (True, "")
    ok, reason = verify_image_file(bad)
    assert not ok and reason


def test_tf_pipeline_is_deterministic_for_eval(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        Image.fromarray(np.full((80, 80), i * 40, dtype=np.uint8)).save(p)
        paths.append(str(p))
    from src.preprocessing import make_tf_dataset

    y = np.zeros((3, 14), dtype=np.float32)
    ds = make_tf_dataset(paths, y, 32, batch_size=2, training=False)
    batches = [x.numpy() for x, _ in ds]
    assert batches[0].shape == (2, 32, 32, 3) and batches[1].shape == (1, 32, 32, 3)
    again = [x.numpy() for x, _ in make_tf_dataset(paths, y, 32, batch_size=2, training=False)]
    assert all(np.array_equal(a, b) for a, b in zip(batches, again))
    train = make_tf_dataset(paths, y, 32, batch_size=2, training=True)
    xb, _ = next(iter(train))
    assert float(xb.numpy().max()) <= 255.0
