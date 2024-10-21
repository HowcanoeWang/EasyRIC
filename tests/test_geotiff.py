import pytest
import pyproj
import re
import tifffile as tf
import numpy as np
import random
import shutil
from pathlib import Path

import easyidp as idp

test_data = idp.data.TestData()
from . import roi_select


def test_def_get_header():
    lotus_full = idp.geotiff.get_header(test_data.pix4d.lotus_dom)
    assert lotus_full["width"] == 5490
    assert lotus_full["height"] == 5752
    assert lotus_full["dim"] == 4
    assert lotus_full["nodata"] == 0
    assert lotus_full["crs"].name == "WGS 84 / UTM zone 54N"
    assert lotus_full["scale"][0] == 0.00738
    assert lotus_full["scale"][1] == 0.00738
    assert lotus_full["tie_point"][0] == 368014.54157
    assert lotus_full["tie_point"][1] == 3955518.2747700005

    lotus_full = idp.geotiff.get_header(test_data.pix4d.lotus_dsm)
    assert lotus_full["width"] == 5490
    assert lotus_full["height"] == 5752
    assert lotus_full["dim"] == 1
    assert lotus_full["nodata"] == -10000.0
    assert lotus_full["crs"].name == "WGS 84 / UTM zone 54N"
    assert lotus_full["scale"][0] == 0.00738
    assert lotus_full["scale"][1] == 0.00738
    assert lotus_full["tie_point"][0] == 368014.54157
    assert lotus_full["tie_point"][1] == 3955518.2747700005

    lotus_part = idp.geotiff.get_header(test_data.pix4d.lotus_dom_part)
    assert lotus_part["width"] == 437
    assert lotus_part["height"] == 444
    assert lotus_part["crs"].name == "WGS 84 / UTM zone 54N"
    assert lotus_part["tie_point"][0] == 368024.0839
    assert lotus_part["tie_point"][1] == 3955479.7512

def test_def_get_imarray():
    maize_part_np = idp.geotiff.get_imarray(test_data.pix4d.maize_dom)
    assert maize_part_np.shape == (722, 836, 4)

    lh = idp.geotiff.get_header(test_data.pix4d.lotus_dom_part)
    lotus_part_np = idp.geotiff.get_imarray(test_data.pix4d.lotus_dom_part)
    assert lotus_part_np.shape == (lh["height"], lh["width"], lh["dim"])


def test_def_geo2pixel2geo_UTM():
    gis_coord = np.asarray([
        [ 484593.67474654, 3862259.42413431],
        [ 484593.41064743, 3862259.92582402],
        [ 484593.64841806, 3862260.06515117],
        [ 484593.93077419, 3862259.55455913],
        [ 484593.67474654, 3862259.42413431]])

    # example file
    # TIFF file: 200423_G_M600pro_transparent_mosaic_group1.tif, 411 MiB, little endian, bigtiff
    # please check v1.0 easyric.tests.test_io_geotiff.py line 114
    # > https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/blob/a3420bc7b1e0f1013411565cf0e66dd2d2ba5371/easyric/tests/test_io_geotiff.py#L114
    # to get the full string of this header
    # here we just use the extracted results
    header = {'width': 19436, 'height': 31255, 'dim':4, 
              'scale': [0.001, 0.001], 'nodata': None,
              'tie_point': [484576.70205, 3862285.5109300003], 
              'crs': pyproj.CRS.from_string("WGS 84 / UTM zone 53N")}

    expected_pixel_idx = np.array([
        [16972, 26086],
        [16708, 25585],
        [16946, 25445],
        [17228, 25956],
        [16972, 26086]])

    expected_pixel_flt = np.array([
        [16972.69654   , 26086.79569047],
        [16708.59742997, 25585.10598028],
        [16946.36805996, 25445.77883044],
        [17228.72418998, 25956.37087012],
        [16972.69654   , 26086.79569047]])

    # ==========================
    # 1. return pixel int index
    # ==========================
    pixel_coord_idx = idp.geotiff.geo2pixel(gis_coord, header, return_index=True)
    np.testing.assert_almost_equal(pixel_coord_idx, expected_pixel_idx)

    # if return index, will cause precision loss
    gis_revert_idx = idp.geotiff.pixel2geo(pixel_coord_idx, header)
    np.testing.assert_almost_equal(gis_revert_idx, gis_coord, decimal=3)

    # ===============================
    # 2. return pixel float position
    # ===============================
    pixel_coord_flt = idp.geotiff.geo2pixel(gis_coord, header)
    np.testing.assert_almost_equal(pixel_coord_flt, expected_pixel_flt)

    # then convert back should have fewer precision loss
    # but seems still have some preoblem
    gis_revert_flt = idp.geotiff.pixel2geo(pixel_coord_idx, header)
    np.testing.assert_almost_equal(gis_revert_flt, gis_coord, decimal=3)


def test_def_geo2pixel2geo_lonlat():
    # using the source: https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/discussions/44
    gis_latlon_coord = np.array([
        [-80.83957435, 25.78354364],
        [-80.83947435, 25.78354364],
        [-80.83947435, 25.78344364],
        [-80.83957435, 25.78344364],
        [-80.83957435, 25.78354364]])

    header = {
        'height': 8748, 'width': 7941, 'dim': 1, 'nodata': -32767.0, 
        'scale': [3.49222000000852e-07, 3.1617399999982425e-07], 
        'tie_point': [-80.84039234705898, 25.784493471936425], 
        'crs': pyproj.CRS.from_epsg(4326)
    }

    expected_pixel = np.array([
        [2342.34114395, 3004.14308711],
        [2628.6919466 , 3004.14308711],
        [2628.6919466 , 3320.42462829],
        [2342.34114395, 3320.42462829],
        [2342.34114395, 3004.14308711]])

    out = idp.geotiff.geo2pixel(gis_latlon_coord, header)

    np.testing.assert_almost_equal(out, expected_pixel)

    back = idp.geotiff.pixel2geo(out, header)

    np.testing.assert_almost_equal(back, gis_latlon_coord, decimal=3)


def test_def_tifffile_crop():
    maize_part_np = idp.geotiff.get_imarray(test_data.pix4d.maize_dom)
    # untiled tiff but 2 row as a patch
    with tf.TiffFile(test_data.pix4d.maize_dom) as tif:
        page = tif.pages[0]

        # even start + even line number
        top = 30
        left = 40
        height = 100
        width = 150

        maize_np_crop = maize_part_np[top:top+height, left:left+width]
        maize_tf_crop = idp.geotiff.tifffile_crop(page, top, left, height, width)

        # vscode debug command to visualize compare two results
        # fig, ax = plt.subplots(1,2); ax[0].imshow(maize_np_crop); ax[1].imshow(maize_tf_crop); ax[0].axis("equal"); ax[1].axis("equal");plt.show()
        np.testing.assert_equal(maize_np_crop, maize_tf_crop)


        # even start + odd line number
        top = 60
        left = 40
        height = 151
        width = 153

        maize_np_crop_odd = maize_part_np[top:top+height, left:left+width]
        maize_tf_crop_odd = idp.geotiff.tifffile_crop(page, top, left, height, width)
        np.testing.assert_equal(maize_np_crop_odd, maize_tf_crop_odd)

        # odd start + even line number:
        top = 91
        left = 40
        height = 230
        width = 153

        maize_np_crop_odd2 = maize_part_np[top:top+height, left:left+width]
        maize_tf_crop_odd2 = idp.geotiff.tifffile_crop(page, top, left, height, width)
        np.testing.assert_equal(maize_np_crop_odd2, maize_tf_crop_odd2)

        # odd start + odd line number
        top = 61
        left = 40
        height = 237
        width = 150

        maize_np_crop_odd2 = maize_part_np[top:top+height, left:left+width]
        maize_tf_crop_odd2 = idp.geotiff.tifffile_crop(page, top, left, height, width)
        np.testing.assert_equal(maize_np_crop_odd2, maize_tf_crop_odd2)


    lotus_part_np = idp.geotiff.get_imarray(test_data.pix4d.lotus_dom_part)
    with tf.TiffFile(test_data.pix4d.lotus_dom_part) as tif:
        page = tif.pages[0]

        top = 30
        left = 40
        height = 100
        width = 150

        lotus_np_crop = lotus_part_np[top:top+height, left:left+width]
        lotus_tf_crop = idp.geotiff.tifffile_crop(page, top, left, height, width)
        np.testing.assert_equal(lotus_np_crop, lotus_tf_crop)


    # read dsm
    lotus_part_np = idp.geotiff.get_imarray(test_data.pix4d.lotus_dsm_part)
    with tf.TiffFile(test_data.pix4d.lotus_dsm_part) as tif:
        page = tif.pages[0]

        top = 30
        left = 40
        height = 100
        width = 150

        lotus_np_crop = lotus_part_np[top:top+height, left:left+width]
        lotus_tf_crop = idp.geotiff.tifffile_crop(page, top, left, height, width)
        np.testing.assert_equal(lotus_np_crop, lotus_tf_crop)


def test_def_point_query():
    # query one point
    point1 = (368023.004, 3955500.669)
    # query one point list
    point2 = [368023.004, 3955500.669]
    # query several points
    point3 = [
        [368022.581, 3955501.054], 
        [368024.032, 3955500.465]]
    # query several points by numpy
    point4 = np.array(point3)

    header = idp.geotiff.get_header(test_data.pix4d.lotus_dsm)
    with tf.TiffFile(test_data.pix4d.lotus_dsm) as tif:
        page = tif.pages[0]

        # point 1
        out1 = idp.geotiff.point_query(page, point1, header)
        expect = np.asarray([97.45558])
        np.testing.assert_almost_equal(out1, expect, decimal=3)

        # point 2
        out2 = idp.geotiff.point_query(page, point2, header)
        np.testing.assert_almost_equal(out2, expect, decimal=3)

        # point 3
        out3 = idp.geotiff.point_query(page, point3, header)
        expects = np.array([97.624344, 97.59617])
        np.testing.assert_almost_equal(out3, expects, decimal=3)

        # point 4
        out4 = idp.geotiff.point_query(page, point4, header)
        np.testing.assert_almost_equal(out4, expects, decimal=3)


def test_def_point_query_raise_error():
    with tf.TiffFile(test_data.pix4d.lotus_dsm) as tif:
        page = tif.pages[0]

        # raise type error
        set1 = {1,2}
        set2 = {1,2,3}

        with pytest.raises(TypeError, match=re.escape("Only tuple, list, ndarray are supported")):
            idp.geotiff.point_query(page, set1)

        with pytest.raises(TypeError, match=re.escape("Only tuple, list, ndarray are supported")):
            idp.geotiff.point_query(page, set2)

        # raise index error
        tuple1 = (1,2,3)
        tuple2 = ((1,2,3), (1,2,3))

        with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
            idp.geotiff.point_query(page, tuple1)

        with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
            idp.geotiff.point_query(page, tuple2)

        list1 = [1,2,3]
        list2 = [[1,2,3],[1,2,3]]

        with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
            idp.geotiff.point_query(page, list1)

        with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
            idp.geotiff.point_query(page, list2)

        ndarray1 = np.array(list1)
        ndarray2 = np.array(list2)

        with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
            idp.geotiff.point_query(page, ndarray1)

        with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
            idp.geotiff.point_query(page, ndarray2)


def test_def_point_query_raise_warn():
    # warn without given header
    point = (3023.004, 3500.669)
    with tf.TiffFile(test_data.pix4d.lotus_dsm) as tif:
        page = tif.pages[0]

        with pytest.warns(UserWarning, match=re.escape("The given pixel coordinates is not integer")):
            out = idp.geotiff.point_query(page, point)

# ==================================================================

# from test_data.pix4d.lotus_dsm
lotus_header_dsm = {'height': 5752, 'width': 5490, 'dim': 1, 
                    'nodata': -10000.0, 'dtype': np.float32, 
                    'scale': [0.00738, 0.00738], 
                    'tie_point': [368014.54157, 3955518.2747700005],
                    'proj': pyproj.CRS.from_epsg(32654)}

# from test_data.pix4d.lotus_dom
lotus_header_dom = {'height': 5752, 'width': 5490, 'dim': 4, 
                    'nodata': 0, 'dtype': np.uint8, 
                    'scale': [0.00738, 0.00738], 
                    'tie_point': [368014.54157, 3955518.2747700005],
                    'proj':pyproj.CRS.from_epsg(32654)}

def test_def_make_is_empty_imarray():
    out1 = idp.geotiff._make_empty_imarray(lotus_header_dsm, 20, 30)
    np.testing.assert_equal(out1, np.ones((20, 30))*-10000.0)

    assert idp.geotiff._is_empty_imarray(lotus_header_dsm, out1)

    rgba = np.ones((20, 30, 4), dtype=np.uint8)*255
    rgba[:,:,3] = rgba[:,:,3] * 0
    lotus_header_dom["dim"] = 3
    out2 = idp.geotiff._make_empty_imarray(lotus_header_dom, 20, 30, layer_num=4)
    np.testing.assert_equal(out2, rgba)

    lotus_header_dom["dim"] = 4
    assert idp.geotiff._is_empty_imarray(lotus_header_dom, out2)

def test_def_make_empty_imarray_error():
    wrong_header1 = {"dim":5, "dtype": np.float32}
    wrong_header2 = {"dim":3, "dtype": np.float32}

    with pytest.raises(ValueError, match=re.escape("Current version only support DSM, RGB and RGBA images (band expect: 1,3,4; get [5], dtype=np.uint8; get [<class 'numpy.float32'>])")):
        idp.geotiff._make_empty_imarray(wrong_header1, 20, 30)

    with pytest.raises(ValueError, match=re.escape("Current version only support DSM, RGB and RGBA images (band expect: 1,3,4; get [3], dtype=np.uint8; get [<class 'numpy.float32'>])")):
        idp.geotiff._make_empty_imarray(wrong_header2, 20, 30)


def test_def_is_empty_imarray_error():
    wrong_header1 = {"dim":5, "dtype": np.float32}
    with pytest.raises(ValueError, match=re.escape("Current version only support DSM, RGB and RGBA images (band expect: 1,3,4; get [5], dtype=np.uint8; get [<class 'numpy.float32'>])")):
        idp.geotiff._is_empty_imarray(wrong_header1, np.zeros((4,4,5)))

    wrong_header2 = {"dim":3, "dtype": np.float32}
    with pytest.raises(IndexError, match=re.escape("The imarray dimention [4] does not match with header dimention [3]")):
        idp.geotiff._is_empty_imarray(wrong_header2, np.zeros((4,4,4)))

def test_def_save_geotiff_and_extratag():
    dom_test = idp.GeoTiff(test_data.tiff.soyweed_part)
    dom_imarray = idp.geotiff.get_imarray(test_data.tiff.soyweed_part)

    left_top_corner = [200, 200]

    extratags = idp.geotiff._offset_geotiff_extratags(dom_test.header, left_top_corner)

    for t in extratags:
        if t[0] == 33922:
            # the key of above function is calculate the correct 
            # <tifffile.TiffTag 33922 ModelTiepointTag @190>
            assert t[3] == (0, 0, 0, 484593.61105, 3862259.86493, 0)

    # test file type error
    with pytest.raises(TypeError, match=re.escape("only *.tif file name is supported")):
        idp.geotiff.save_geotiff(dom_test.header, dom_imarray, left_top_corner, "random_name.tifx")

    # test whether correctly saved
    save_tiff = test_data.tiff.out / "def_save_geotiff_test.tif"
    if save_tiff.exists():
        save_tiff.unlink()

    idp.geotiff.save_geotiff(dom_test.header, dom_imarray, left_top_corner, save_tiff)

    assert save_tiff.exists()

    # check if successfully offseted 20x20 pixels
    # visually check the output and input DOM in QGIS no problem
    # here use code to check
    saved_dom = idp.GeoTiff(save_tiff)

    assert saved_dom.header['tie_point'] == [484593.61105, 3862259.86493]
    assert dom_test.header['tie_point'] == [484593.41105, 3862260.0649300003]


# ==============
# GeoTiff Class
# ==============

# This should be activated after geotiff.save_geotiff() function totally finished
# def test_class_check_hasfile_decorator():
#     obj = idp.GeoTiff()

#     with pytest.raises(FileNotFoundError, match=re.escape("Could not operate if not specify correct geotiff file")):
#         obj.save_geotiff(np.ones((3,3)), np.ones((1,2)), "wrong_path")

#     obj2 = idp.GeoTiff(test_data.pix4d.lotus_dom)
#     obj2.file_path = f"not/exists/path"
#     with pytest.raises(FileNotFoundError, match=re.escape("Could not operate if not specify correct geotiff file")):
#         obj2.save_geotiff(np.ones((3,3)), np.ones((1,2)), "wrong_path")

#     obj3 = idp.GeoTiff(test_data.pix4d.lotus_dom)
#     obj3.header = None
#     with pytest.raises(FileNotFoundError, match=re.escape("Could not operate if not specify correct geotiff file")):
#         obj3.save_geotiff(np.ones((3,3)), np.ones((1,2)), "wrong_path")

def test_class_init_with_path():
    obj = idp.GeoTiff(test_data.pix4d.lotus_dom)

    # convert rel path to abs path, ideally it should longer
    assert Path(obj.file_path).resolve() == test_data.pix4d.lotus_dom.resolve()
    assert obj.header is not None

def test_class_header_sugar_property():
    # test the crs sugar to replace geotiff.header['crs']
    obj = idp.GeoTiff(test_data.pix4d.lotus_dom)
    
    assert obj.crs == obj.header['crs']
    assert obj.height == obj.header['height']
    assert obj.width == obj.header['width']
    assert obj.dim == obj.header['dim']
    assert obj.nodata == obj.header['nodata']
    assert obj.scale == obj.header['scale']
    assert obj.tie_point == obj.header['tie_point']

    # test value setter
    with pytest.raises(AttributeError):
        # python <3.10 : can't set attribute ...
        # python >3.10 : property 'crs' of 'GeoTiff' object has no setter
        obj.crs = 'aaa'


def test_class_read_geotiff():
    obj = idp.GeoTiff()
    obj.read_geotiff(test_data.pix4d.lotus_dom)

    assert Path(obj.file_path).resolve() == test_data.pix4d.lotus_dom.resolve()
    assert obj.header is not None

def test_class_crop_polygon_save_geotiff():
    obj = idp.GeoTiff(test_data.pix4d.lotus_dom)

    plot, proj = idp.shp.read_shp(test_data.shp.lotus_shp, name_field=0, return_proj=True)
    plot_t = idp.geotools.convert_proj(plot, proj, obj.header["crs"])

    # test case 1
    #  polygon_hv = plot_t["N1W1"]
    # then do random choose
    plot_id, polygon_hv = random.choice(list(plot_t.items()))

    save_tiff = test_data.tiff.out / "crop_polygon.tif"
    if save_tiff.exists():
        save_tiff.unlink()
    imarray = obj.crop_polygon(polygon_hv, is_geo=True, save_path=save_tiff)

    assert save_tiff.exists()
    #assert imarray.shape == (320, 319, 4)  # N1W1
    assert len(imarray.shape) == 3  # like (m, n, d)
    assert imarray.shape[2] == 4  # d = 4
    # around 300 pixels for all squared lotus boundary
    assert 270 < imarray.shape[0] and imarray.shape[0] < 350
    assert 270 < imarray.shape[1] and imarray.shape[1] < 350

    out = idp.GeoTiff(save_tiff)
    '''
    N1W1 case, the output offset are not the same as input
    ------------------------------------------------------
    result ->
        [368017.75187, 3955511.4999300004]
    polygon_hv (input) ->
        min -> [368017.7565143, 3955509.13563382]
        max -> [368020.11263046, 3955511.49811902]
      x ~= min, y ~= max
    offset_in_func (same as result)
        roi_offset(pixel) -> [435, 918]
        roi_offset(pixel2geo) -> [ 368017.75187, 3955511.49993]
    ----------------------
    reason
    A         B  -> pixel edge
    |---o-----|--
    |   |     |--
    |   polygon points (polygon_hv)
    crop -> use pixel edge as offset

    A = result
    B = result + scale
    A <= polygon_hv.minmax <= B
    '''
    xmin, _ = polygon_hv.min(axis=0)
    _, ymax = polygon_hv.max(axis=0)

    assert xmin >= out.header["tie_point"][0]
    assert xmin <= out.header["tie_point"][0] + out.header["scale"][0]
    assert ymax <= out.header["tie_point"][1]
    assert ymax >= out.header["tie_point"][1] - out.header["scale"][1]


def test_crop_rectange_save_geotiff():
    obj = idp.GeoTiff(test_data.pix4d.lotus_dom)

    out1 = obj.crop_rectangle(left=434, top=918, w=320, h=321, is_geo=False)

    out2 = obj.crop_rectangle(
        left=368017.75187, top=3955511.49993, 
        w=2.3561161599936895, h=2.362485199701041, 
        is_geo=True)

    assert out1.shape == (321, 320, 4)
    assert out2.shape == (321, 320, 4)

    np.testing.assert_almost_equal(out1, out2)

    # check if 
    with pytest.raises(IndexError, match=re.escape(
        f"The given rectange [left 368017.75187, top 3955511.49993, "
        f"width 2.3561161599936895, height 2.362485199701041] can not fit "
        f"into geotiff shape [0, 0, 5490, 5752]. ")):
        out2 = obj.crop_rectangle(
        left=368017.75187, top=3955511.49993, 
        w=2.3561161599936895, h=2.362485199701041, 
        is_geo=False)


def test_class_polygon_math():
    # test dsm results
    dsm = idp.GeoTiff(test_data.pix4d.lotus_dsm)

    # plot_t["N1W1"] -> 
    poly_geo = np.array([
        [ 368017.7565143 , 3955511.08102277],
        [ 368019.70190232, 3955511.49811902],
        [ 368020.11263046, 3955509.54636219],
        [ 368018.15769062, 3955509.13563382],
        [ 368017.7565143 , 3955511.08102277]])

    dsm_mean   = dsm.polygon_math(poly_geo, is_geo=True, kernel="mean")
    dsm_min    = dsm.polygon_math(poly_geo, is_geo=True, kernel="min")
    dsm_max    = dsm.polygon_math(poly_geo, is_geo=True, kernel="max")
    dsm_pmin5  = dsm.polygon_math(poly_geo, is_geo=True, kernel="pmin5")
    dsm_pmin10 = dsm.polygon_math(poly_geo, is_geo=True, kernel="pmin10")
    dsm_pmax5  = dsm.polygon_math(poly_geo, is_geo=True, kernel="pmax5")
    dsm_pmax10 = dsm.polygon_math(poly_geo, is_geo=True, kernel="pmax10")

    assert 97 < dsm_mean   and dsm_mean   < 98
    assert 97 < dsm_min    and dsm_min    < 98
    assert 97 < dsm_max    and dsm_max    < 98
    assert 97 < dsm_pmin5  and dsm_pmin5  < 98
    assert 97 < dsm_pmin10 and dsm_pmin10 < 98
    assert 97 < dsm_pmax5  and dsm_pmax5  < 98
    assert 97 < dsm_pmax10 and dsm_pmax10 < 98

    # test dom results
    dom = idp.GeoTiff(test_data.pix4d.lotus_dom)

    dom_mean   = dom.polygon_math(poly_geo, is_geo=True, kernel="mean")
    dom_min    = dom.polygon_math(poly_geo, is_geo=True, kernel="min")
    dom_max    = dom.polygon_math(poly_geo, is_geo=True, kernel="max")
    dom_pmin5  = dom.polygon_math(poly_geo, is_geo=True, kernel="pmin5")
    dom_pmin10 = dom.polygon_math(poly_geo, is_geo=True, kernel="pmin10")
    dom_pmax5  = dom.polygon_math(poly_geo, is_geo=True, kernel="pmax5")
    dom_pmax10 = dom.polygon_math(poly_geo, is_geo=True, kernel="pmax10")

    assert dom_mean  .shape == (4, )
    assert dom_min   .shape == (4, )
    assert dom_max   .shape == (4, )
    assert dom_pmin5 .shape == (4, )
    assert dom_pmin10.shape == (4, )
    assert dom_pmax5 .shape == (4, )
    assert dom_pmax10.shape == (4, )

    assert dom_mean  [3] == 255.0
    assert dom_min   [3] == 255.0
    assert dom_max   [3] == 255.0
    assert dom_pmin5 [3] == 255.0
    assert dom_pmin10[3] == 255.0
    assert dom_pmax5 [3] == 255.0
    assert dom_pmax10[3] == 255.0

def test_class_point_query():
    obj = idp.GeoTiff(test_data.pix4d.lotus_dsm)

    # plot_t["N1W1"] -> 
    poly_geo = np.array([
        [ 368017.7565143 , 3955511.08102277],
        [ 368019.70190232, 3955511.49811902],
        [ 368020.11263046, 3955509.54636219],
        [ 368018.15769062, 3955509.13563382],
        [ 368017.7565143 , 3955511.08102277]])

    pt = obj.point_query(poly_geo, is_geo=True)

    assert pt.shape == (5,)
    assert np.all(97 < pt) and np.all(pt < 98)

def test_class_crop_rois():
    obj = idp.GeoTiff(test_data.pix4d.lotus_dom)

    roi = roi_select.copy()

    roi.get_z_from_dsm(test_data.pix4d.lotus_dsm, mode="point", kernel="mean", buffer=0, keep_crs=False)

    tif_out_folder = test_data.tiff.out / "class_crop"
    if tif_out_folder.exists():
        shutil.rmtree(tif_out_folder)
    tif_out_folder.mkdir()

    out_dict = obj.crop_rois(roi, save_folder=tif_out_folder)

    assert len(out_dict) == 4
    assert (tif_out_folder / "N1W1.tif").exists()
    assert out_dict["N2E2"].shape == (320, 320, 4)

def test_class_crop_rois_multispec():
    roi = idp.ROI(test_data.shp.mlayer_shp)

    # 5 layers multispectral with 5th as alpha
    multi_tiff = idp.GeoTiff(test_data.tiff.mlayer_multi)
    multi_tiff.transparent_layer = -1

    tif_out_folder = test_data.tiff.out / "multi_crop"
    if tif_out_folder.exists():
        shutil.rmtree(tif_out_folder)
    tif_out_folder.mkdir()

    out_dict = multi_tiff.crop_rois(roi, save_folder=tif_out_folder)
    assert len(out_dict) == 12
    assert (tif_out_folder / "2.tif").exists()
    assert out_dict["2"].shape == (180, 180, 5)


def test_class_crop_rois_ndvi_special():
    roi = idp.ROI(test_data.shp.mlayer_shp)
    # processing multispectral geotiff -> 2 layer ndvi (second as alpha)
    ndvi_tiff = idp.GeoTiff(test_data.tiff.mlayer_ndvi)

    tif_out_folder = test_data.tiff.out / "ndvi_crop"
    if tif_out_folder.exists():
        shutil.rmtree(tif_out_folder)
    tif_out_folder.mkdir()

    out_dict = ndvi_tiff.crop_rois(roi, save_folder=tif_out_folder)
    assert len(out_dict) == 12
    assert (tif_out_folder / "2.tif").exists()
    assert out_dict["2"].shape == (180, 180, 2)


def test_class_geo2pixel2geo_executable():

    roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
    dom = idp.GeoTiff(test_data.pix4d.lotus_dom)
    roi.change_crs(dom.header['crs'])

    roi_test = roi[111]

    roi_test_pixel = dom.geo2pixel(roi_test)

    roi_test_back = dom.pixel2geo(roi_test_pixel)

    np.testing.assert_almost_equal(roi_test, roi_test_back, decimal=5)