from collections import Counter


def test_subject_grouped_batches_keep_same_subject_nights_together():
    from training.train_tso_patch_h5 import subject_grouped_batch_generator

    class _Stub:
        def __init__(self, subjects):
            self.subjects = subjects

        def __len__(self):
            return len(self.subjects)

    subjects = ["S1", "S1", "S1", "S2", "S2", "S3", "S3", "S3", "S3"]
    ds = _Stub(subjects)

    seen = []
    positive_batches = 0
    total_batches = 0
    for batch in subject_grouped_batch_generator(ds, batch_size=4, nights_per_group=4):
        positions = batch.tolist()
        seen.extend(positions)
        counts = Counter(subjects[p] for p in positions)
        total_batches += 1
        if any(c >= 2 for c in counts.values()):
            positive_batches += 1

    assert sorted(seen) == list(range(len(subjects)))  # every night emitted exactly once
    assert positive_batches == total_batches            # every batch has SupCon positives
