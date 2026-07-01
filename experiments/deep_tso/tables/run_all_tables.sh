#!/usr/bin/env bash
# run_all_tables.sh — train every arm for Tables 1-6 (shared arms train once thanks to
# run_arm's skip-if-exists), then aggregate. ~14 distinct cross-dataset arms; sequence it.
# Override GPU/paths via env (see _common.sh). Consistency arm forces 4-GPU internally.
set -euo pipefail
here="$(dirname "$0")"

bash "$here/table2_noisy_label.sh"          # baseline + gce family + structural (covers Table 1/3 CE too)
bash "$here/table3_crossnight_supcon.sh"    # + ce_supcon
bash "$here/table1_main_results.sh"         # (baseline + ce_supcon already done)
bash "$here/table4_architecture.sh"         # e1_* arms
bash "$here/table6_structured_output.sh"    # + interval
bash "$here/table5_crossnight_consistency.sh"   # + consistency

echo
echo "All arms done. Aggregating ->"
bash "$here/inspect.sh"
