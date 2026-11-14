import tempfile
import unittest
from pathlib import Path

from tts.chatterbox_compat import EXPECTED_T3_MODEL, loader_model_dir


class LoaderModelDirTests(unittest.TestCase):
    def test_aliases_telugu_checkpoint_for_stock_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            model_dir.mkdir()
            (model_dir / EXPECTED_T3_MODEL).write_bytes(b"telugu-weights")
            (model_dir / "conds.pt").write_bytes(b"conds")
            (model_dir / "nested").mkdir()
            (model_dir / "nested" / "ignored.txt").write_text("ignored", encoding="utf-8")

            with loader_model_dir(model_dir) as loader_dir:
                self.assertNotEqual(loader_dir, model_dir)
                alias = loader_dir / "t3_mtl23ls_v2.safetensors"
                self.assertTrue(alias.is_file())
                self.assertEqual(alias.read_bytes(), b"telugu-weights")
                self.assertEqual((loader_dir / "conds.pt").read_bytes(), b"conds")
                self.assertFalse((loader_dir / "nested").exists())
                self.assertTrue((model_dir / EXPECTED_T3_MODEL).is_file())
                self.assertFalse((model_dir / "t3_mtl23ls_v2.safetensors").exists())
                loader_path = loader_dir

            self.assertFalse(loader_path.exists())

    def test_without_telugu_checkpoint_returns_original_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            model_dir.mkdir()
            (model_dir / "conds.pt").write_bytes(b"conds")

            with loader_model_dir(model_dir) as loader_dir:
                self.assertEqual(loader_dir, model_dir)
                self.assertEqual((loader_dir / "conds.pt").read_bytes(), b"conds")

    def test_custom_alias_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            model_dir.mkdir()
            (model_dir / EXPECTED_T3_MODEL).write_bytes(b"telugu-weights")

            with loader_model_dir(model_dir, alias_filename="checkpoint.safetensors") as loader_dir:
                self.assertEqual(
                    (loader_dir / "checkpoint.safetensors").read_bytes(),
                    b"telugu-weights",
                )
                self.assertFalse((loader_dir / "t3_mtl23ls_v2.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
