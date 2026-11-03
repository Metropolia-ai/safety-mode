# Warehouse test video

Three clips from Pexels (Pexels license: free, no attribution, may be modified; may not be sold unmodified, and identifiable people may not be shown as endorsing anything). The files are not in git: `./download.sh` puts them in `raw/`. The bench itself cuts one frame per second (ffmpeg) into a temporary folder.

| File | Source | Length | What is visible | Rules to check |
|---|---|---|---|---|
| pexels-32243653-forklift-operator.mp4 | https://www.pexels.com/video/warehouse-worker-operating-forklift-safely-32243653/ (Usman AbdulrasheedGambo) | 10 s, 2560×1440, 25 fps | close-up: operator in a red hard hat, no vest, lifts a pallet; no people nearby | 2 (hard hat yes, vest no → alert if the "past the gate" zone covers the frame), 3 (no alert) |
| pexels-15170997-warehouse-team.mp4 | https://www.pexels.com/video/workers-at-a-warehouse-15170997/ (Rolan Muradov) | 22 s, 1080×1920 vertical, 30 fps | three workers carry a truss next to a reach truck: two in orange vests and hard hats, one in a black jacket without a vest; racks, flat floor | 2 (one person without a vest → alert; two in PPE → none), 3 (reach truck next to people), 6 (racks in frame, no climbing → no alert) |
| pexels-6079421-forklift-busy-warehouse.mp4 | https://www.pexels.com/video/person-driving-forklift-6079421/ (K / @kelly) | 25 s, 2560×1440, 24 fps | a Toyota forklift with an operator without a hard hat drives along an aisle, a pedestrian by the gate in the background, a yellow line along the aisle, racks | 3 (forklift–pedestrian in the far background: distance check), 2 (operator without a hard hat), 5 (gate as fire_exit_zone), 8 (yellow line as the lane boundary) |

Expected per-second results for each rule (where an alert must fire and where it must not) are in `../../test-footage/expected.md`; zones and floor calibration, marked by hand on the first frame of each clip, are in `../../test-footage/zones/<name>.json`.

A PPE violation is already covered (clip 2: a person without a vest; clip 3: an operator without a hard hat). Missing: a person lying down (rule 10) and a person inside a marked zone next to a running machine (rule 1).
