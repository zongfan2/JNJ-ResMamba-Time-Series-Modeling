# -*- coding: utf-8 -*-
"""Count nights (segments) per subject for the UKB TSO data.

The cross-night regularizers in train_tso_patch_h5.py group by SUBJECT and only
fire when a subject contributes >=2 nights to a batch:
  - SupCon (C3) needs >=2 same-subject nights to form a positive pair;
  - cross_night_consistency_loss (C3') returns 0 unless a subject has >=2 nights.
So if UKB is (mostly) one night per subject, both arms are no-ops on UKB and the
in-domain noprod data is the place to test them. This script measures that.

It mirrors the trainer's grouping key EXACTLY:
  * from raw parquet filenames  -> parse_biobank_filename(): subject = parts[1]
    (e.g. Processed_<eid>_<eid>_<instance>_0_0_<date>.parquet.gzip -> <eid>)
  * from a built H5             -> _subject_from_segment_name(segment_name)
    = segment_name.split('_')[0]  (segment_name = "<subject>_<wrist>_<instance>_<day>")
Both reduce to the same subject id, so the two modes agree.

Usage (Domino):
  # before/without a built H5 -- scan the raw parquet folder:
  python3.11 test-tools/check_ukb_nights_per_subject.py \
      --input_folder /mnt/imported/data/NocturnalScratch_Analysis/UKB_v2/raw/

  # after building the supervised TSO H5 -- read it back:
  python3.11 test-tools/check_ukb_nights_per_subject.py \
      --h5 /mnt/data/GENEActive-featurized/h5/ukb_20hz_sincos.h5
"""

import argparse
import glob
import os
from collections import Counter


def subject_from_filename(path):
    """Replicates training/convert_h5.py parse_biobank_filename() subject id."""
    basename = os.path.basename(path).replace('.parquet.gzip', '').replace('.parquet', '')
    parts = basename.split('_')
    if len(parts) >= 7 and parts[0] == 'Processed':
        return parts[1], parts[-1]            # (subject, day)
    return (parts[1] if len(parts) > 1 else 'unknown',
            parts[-1] if parts else 'unknown')


def subject_from_segment_name(name):
    """Replicates training/train_tso_patch_h5.py _subject_from_segment_name()."""
    if isinstance(name, bytes):
        name = name.decode('utf-8', 'replace')
    name = str(name)
    return name.split('_')[0] if '_' in name else name


def summarize(seg_subjects, seg_days=None):
    """seg_subjects: list of subject ids, one per segment (night)."""
    n_segments = len(seg_subjects)
    per_subject = Counter(seg_subjects)              # segments (nights) per subject
    counts = sorted(per_subject.values())
    n_subjects = len(per_subject)

    multi = {s: c for s, c in per_subject.items() if c >= 2}
    n_multi = len(multi)
    segs_in_multi = sum(c for c in per_subject.values() if c >= 2)

    def pct(x):
        return 100.0 * x / n_segments if n_segments else 0.0

    print(f"Segments (nights) total : {n_segments}")
    print(f"Unique subjects         : {n_subjects}")
    if n_subjects:
        mean = n_segments / n_subjects
        median = counts[len(counts) // 2]
        print(f"Nights/subject          : min {counts[0]}  median {median}  "
              f"mean {mean:.2f}  max {counts[-1]}")
    multi_pct = (100.0 * n_multi / n_subjects) if n_subjects else 0.0
    print(f"Subjects with >=2 nights : {n_multi}/{n_subjects} ({multi_pct:.1f}%)")
    print(f"Segments in >=2-night subj: {segs_in_multi}/{n_segments} ({pct(segs_in_multi):.1f}%)")

    # Distribution histogram of nights-per-subject.
    hist = Counter(counts)
    print("\nNights-per-subject distribution (nights: #subjects):")
    for k in sorted(hist):
        bar = '#' * min(hist[k], 50)
        print(f"  {k:>3} night(s): {hist[k]:>6}  {bar}")

    if seg_days is not None:
        # distinct calendar dates per subject (true nights, in case a subject has
        # multiple files for the same date, e.g. two wrists).
        by_subj_days = {}
        for s, d in zip(seg_subjects, seg_days):
            by_subj_days.setdefault(s, set()).add(d)
        distinct = sorted(len(v) for v in by_subj_days.values())
        if distinct and distinct[-1] != counts[-1]:
            print(f"\n(distinct DATES/subject: min {distinct[0]} median "
                  f"{distinct[len(distinct)//2]} max {distinct[-1]} — differs from "
                  f"segments/subject, so some subjects have multiple files per date)")

    print("\nVerdict:")
    if segs_in_multi == 0:
        print("  ALL subjects have exactly 1 night -> cross-night SupCon AND consistency"
              "\n  are NO-OPS on this data. Test C3/C3' on noprod instead.")
    elif pct(segs_in_multi) < 25:
        print(f"  Only {pct(segs_in_multi):.0f}% of segments are in multi-night subjects ->"
              "\n  the cross-night terms fire rarely; expect a weak/near-zero contribution.")
    else:
        print(f"  {pct(segs_in_multi):.0f}% of segments are in multi-night subjects ->"
              "\n  the cross-night terms have enough positives to contribute.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--input_folder', help='Folder of raw UKB *.parquet(.gzip) files.')
    g.add_argument('--h5', help='A built supervised TSO H5 (reads /segment_names).')
    args = ap.parse_args()

    if args.input_folder:
        files = sorted(glob.glob(os.path.join(args.input_folder, '*.parquet.gzip'))
                       + glob.glob(os.path.join(args.input_folder, '*.parquet')))
        if not files:
            raise SystemExit(f"No parquet files under {args.input_folder!r}")
        print(f"Source: {len(files)} parquet files in {args.input_folder}\n")
        parsed = [subject_from_filename(f) for f in files]
        summarize([s for s, _ in parsed], [d for _, d in parsed])
    else:
        import h5py
        with h5py.File(args.h5, 'r') as f:
            n = int(f.attrs.get('num_segments', f['segment_names'].shape[0]))
            names = f['segment_names'][:n]
            has_subj = 'subject_ids' in f
            subj = ([s.decode() if isinstance(s, bytes) else str(s)
                     for s in f['subject_ids'][:n]] if has_subj
                    else [subject_from_segment_name(nm) for nm in names])
        print(f"Source: H5 {args.h5}  (num_segments={n}, "
              f"subject_ids={'present' if has_subj else 'derived from segment_names'})\n")
        summarize(subj)


if __name__ == '__main__':
    main()
