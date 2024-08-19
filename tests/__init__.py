import sys
from pathlib import Path
import easyidp as idp

sys.path.insert(0, ".")

# check if output path exists
out_dir = Path("./tests/out")
if not out_dir.exists():
    out_dir.mkdir()

out_folders = [
    "json_test", "pcd_test", "cv_test", "tiff_test", 
    "visual_test", "back2raw_test", "data_test"
]

for o in out_folders:
    sub_dir = out_dir / o
    if not sub_dir.exists():
        sub_dir.mkdir()

test_data = idp.data.TestData()

roi_all = idp.ROI(test_data.shp.lotus_shp, name_field=0)

# global variable for testing
# shorten for quick for loops
roi_select = idp.ROI() 
for key in ["N1W1", "N1W2", "N2E2", "S1W1"]:
    roi_select[key] = roi_all[key]
    roi_select.crs = roi_all.crs
    roi_select.source = roi_all.source

