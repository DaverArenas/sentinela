# Test Fixtures

Vision tests run against **recorded video**, so every run is repeatable and needs no camera
or internet connection.

- Synthetic clips are generated at test time (`tests/support.py::write_video`) and written to
  pytest's temporary directory; they cover ingestion, sampling, reconnect and event flow.
- Real footage for acceptance tests (for example "a person enters a configured zone") goes in
  `tests/fixtures/videos/` **only** if it is non-private, consented, and licensed for this use.
  Record its origin and license next to the file. Video files are git-ignored by default;
  never commit private or customer footage.
- Model-backed tests are marked `model` and skip automatically when `models/yolox_s.onnx` is absent.
