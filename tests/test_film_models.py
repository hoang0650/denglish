import catalog
import handler
import prefetch_volume


def test_film_models_are_allowlisted_and_routed():
    assert handler.model_for("i2v")["key"] == "wan2.2-i2v-a14b"
    assert handler.model_for("image_edit")["key"] == "qwen-image-edit-2511"
    assert handler._TASK["i2v"] == "i2v" and handler._TASK["image_edit"] == "image_edit"
    for key in ("wan2.2-i2v-a14b", "qwen-image-edit-2511"):
        row = catalog.MODELS[key]
        assert row["license"] == "Apache-2.0" and row["lora"]["license"] == "Apache-2.0"
        assert row["manual"] and not row["default"]


def test_prefetch_never_pulls_manual_models(tmp_path):
    out = prefetch_volume.prefetch(["wan2.2-i2v-a14b", "qwen-image-edit-2511"], root=str(tmp_path))
    assert {r["status"] for r in out["models"].values()} == {"manual"}


def test_fit_720p_sizes():
    Image = __import__("pytest").importorskip("PIL.Image")

    assert handler._fit_720p(Image.new("RGB", (1920, 1080)))[1] == (1280, 720)
    assert handler._fit_720p(Image.new("RGB", (1080, 1920)))[1] == (720, 1280)
    assert handler._fit_720p(Image.new("RGB", (1000, 1000)))[1] == (960, 960)
