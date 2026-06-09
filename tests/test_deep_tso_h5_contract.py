import h5py
import numpy as np


def test_h5_dataset_reads_subject_ids_and_falls_back_to_segment_prefix(tmp_path):
    from training.train_tso_patch_h5 import H5Dataset

    path = tmp_path / "mini.h5"
    with h5py.File(path, "w") as h5f:
        h5f.attrs["num_segments"] = 2
        h5f.attrs["max_seq_length"] = 120
        h5f.attrs["samples_per_second"] = 20
        h5f.attrs["max_len"] = 120
        h5f.attrs["num_channels"] = 6
        h5f.create_dataset("X", data=np.zeros((2, 120, 6), dtype=np.float32))
        h5f.create_dataset("Y", data=np.zeros((2, 120, 2), dtype=np.int8))
        h5f.create_dataset("seq_lengths", data=np.array([120, 120], dtype=np.int32))
        h5f.create_dataset("segment_names", data=np.array(["S1_0_day1", "S2_0_day1"], dtype=h5py.string_dtype()))

    ds = H5Dataset(str(path))
    assert ds[0]["subject"] == "S1"
    assert ds[1]["subject"] == "S2"
    ds.close()


def test_h5_dataset_reads_annotator_tracks_when_present(tmp_path):
    from training.train_tso_patch_h5 import H5Dataset

    path = tmp_path / "mini_annotators.h5"
    with h5py.File(path, "w") as h5f:
        h5f.attrs["num_segments"] = 1
        h5f.attrs["max_seq_length"] = 120
        h5f.attrs["samples_per_second"] = 20
        h5f.attrs["max_len"] = 120
        h5f.attrs["num_channels"] = 6
        h5f.attrs["annotator_names"] = np.array(["sadeh", "cole_kripke"], dtype=h5py.string_dtype())
        h5f.create_dataset("X", data=np.zeros((1, 120, 6), dtype=np.float32))
        h5f.create_dataset("Y", data=np.zeros((1, 120, 2), dtype=np.int8))
        h5f.create_dataset("Y_annotators", data=np.ones((1, 120, 2), dtype=np.int8))
        h5f.create_dataset("subject_ids", data=np.array(["S1"], dtype=h5py.string_dtype()))
        h5f.create_dataset("seq_lengths", data=np.array([120], dtype=np.int32))
        h5f.create_dataset("segment_names", data=np.array(["S1_0_day1"], dtype=h5py.string_dtype()))

    ds = H5Dataset(str(path))
    sample = ds[0]
    assert sample["Y_annotators"].shape == (120, 2)
    assert sample["subject"] == "S1"
    ds.close()
