"""Optional differential check against the ACTUAL original C KD-tree library.

No ROS/Eigen needed. A C compiler is optional; absent compiler/source => skip.
The temporary shared library is built outside both source folders.
"""

import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

from path_finder.kdtree import KDTree
from path_finder.planners import Node


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / "sampling-based-path-finding-main"


class SourceProvenanceTests(unittest.TestCase):
    @unittest.skipUnless(ORIGINAL.exists(), "Sibling C++ source is not available")
    def test_original_source_hashes_unchanged(self):
        manifest = json.loads((ROOT / "source_manifest.json").read_text())
        for relative, expected in manifest.items():
            with self.subTest(source=relative):
                self.assertEqual(hashlib.sha256((ROOT.parent / relative).read_bytes()).hexdigest(), expected)


@unittest.skipUnless(shutil.which("cc") and ORIGINAL.exists(), "Optional C compiler/source not available")
class NativeKDTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="dacn-kdtree-")
        library = Path(cls.temporary.name) / ("kd.dylib" if sys.platform == "darwin" else "kd.so")
        source = ORIGINAL / "src/path_finder/src/kdtree.c"
        include = ORIGINAL / "src/path_finder/include"
        subprocess.run([shutil.which("cc"), "-shared", "-fPIC", "-O2", "-I", str(include),
                        str(source), "-o", str(library)], check=True, capture_output=True)
        cls.lib = C.CDLL(str(library))
        cls.lib.kd_create.argtypes, cls.lib.kd_create.restype = [C.c_int], C.c_void_p
        cls.lib.kd_insert3.argtypes = [C.c_void_p, C.c_double, C.c_double, C.c_double, C.c_void_p]
        cls.lib.kd_nearest3.argtypes = [C.c_void_p, C.c_double, C.c_double, C.c_double]
        cls.lib.kd_nearest3.restype = C.c_void_p
        cls.lib.kd_nearest_range3.argtypes = [C.c_void_p, C.c_double, C.c_double, C.c_double, C.c_double]
        cls.lib.kd_nearest_range3.restype = C.c_void_p
        cls.lib.kd_res_item_data.argtypes, cls.lib.kd_res_item_data.restype = [C.c_void_p], C.c_void_p
        cls.lib.kd_res_end.argtypes, cls.lib.kd_res_end.restype = [C.c_void_p], C.c_int
        cls.lib.kd_res_next.argtypes = [C.c_void_p]
        cls.lib.kd_res_free.argtypes = [C.c_void_p]
        cls.lib.kd_free.argtypes = [C.c_void_p]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_incremental_nearest_and_ordered_range_results_match_original_c(self):
        rng, py = random.Random(291), KDTree()
        native = self.lib.kd_create(3)
        try:
            for i in range(400):
                n = Node(i + 1, (rng.uniform(-10, 10), rng.uniform(-10, 10), 0), 0)
                py.insert(n)
                self.assertEqual(self.lib.kd_insert3(native, *n.x, C.c_void_p(n.id)), 0)
                if i % 4:
                    continue
                p = rng.uniform(-10, 10), rng.uniform(-10, 10), 0
                found = self.lib.kd_nearest3(native, *p)
                try:
                    self.assertEqual(py.nearest(p).id, self.lib.kd_res_item_data(found))
                finally:
                    self.lib.kd_res_free(found)
                found = self.lib.kd_nearest_range3(native, *p, 5.0)
                try:
                    ids = []
                    while not self.lib.kd_res_end(found):
                        ids.append(self.lib.kd_res_item_data(found))
                        self.lib.kd_res_next(found)
                    self.assertEqual([n.id for n in py.within(p, 5.0)], ids)
                finally:
                    self.lib.kd_res_free(found)
        finally:
            self.lib.kd_free(native)


if __name__ == "__main__":
    unittest.main()
