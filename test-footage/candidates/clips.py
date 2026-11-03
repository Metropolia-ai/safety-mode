"""Candidate clips: source, trim, licence. Shared by prepare.py and run_candidates.py."""
CLIPS = {
    "pexels-38707991-man-crosses-floor": {"id": 38707991, "url": "https://www.pexels.com/video/warehouse-storage-with-worker-walking-38707991/", "trim": None},
    "pexels-4280746-hatless-visitor": {"id": 4280746, "url": "https://www.pexels.com/video/men-talking-while-walking-in-the-warehouse-4280746/", "trim": None},
    "pexels-31751344-forklift-crosses-aisle": {"id": 31751344, "url": "https://www.pexels.com/video/warehouse-forklift-moving-through-storage-aisles-31751344/", "trim": (0, 12)},
    "pexels-32245123-site-worker-no-vest": {"id": 32245123, "url": "https://www.pexels.com/video/construction-workers-operating-on-building-site-32245123/", "trim": None},
    "pexels-32268407-forklift-roll-worker": {"id": 32268407, "url": "https://www.pexels.com/video/efficient-forklift-operation-in-industrial-warehouse-32268407/", "trim": (0, 43)},
    "pexels-13422071-scrubber-worker": {"id": 13422071, "url": "https://www.pexels.com/video/workers-cleaning-warehouse-13422071/", "trim": None},
}
SAM_FPS = 10   # video sent to SAM 3.1 is resampled to 10 fps: billing is per frame
