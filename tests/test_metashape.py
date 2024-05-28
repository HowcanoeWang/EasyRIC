import re
import pytest
import pyproj
import numpy as np
import xml.etree.ElementTree as ET

import easyidp as idp

test_data = idp.data.TestData()
from . import roi_select

#########################
# test math calculation #
#########################
def test_apply_transform_matrix():
    matrix = np.asarray([
        [-0.86573098, -0.01489186,  0.08977677,  7.65034123],
        [ 0.06972335,  0.44334391,  0.74589315,  1.85910928],
        [-0.05848325,  0.74899678, -0.43972184, -0.1835615],
        [ 0.,          0.,          0.,          1.]], dtype=float)
    matrix2 = np.linalg.inv(matrix)

    point1 = np.array([0.5, 1, 1.5], dtype=float)
    out1 = idp.metashape.apply_transform_matrix(point1, matrix2)

    ans1 = np.array([7.96006409,  1.30195288, -2.66971818])
    np.testing.assert_array_almost_equal(out1, ans1, decimal=6)

    np_array = np.array([
        [0.5, 1, 1.5],
        [0.5, 1, 1.5]], dtype=float)
    out2 = idp.metashape.apply_transform_matrix(np_array, matrix2)
    ans2 = np.array([
        [7.96006409, 1.30195288, -2.66971818],
        [7.96006409, 1.30195288, -2.66971818]])
    np.testing.assert_array_almost_equal(out2, ans2, decimal=6)


def test_convert_proj3d():
    # transform between geocentric and geodetic
    geocentric = np.array([-3943658.7087006606, 3363404.124223561, 3704651.3067566575])

    # columns=['lon', 'lat', 'alt']
    geodetic = np.array([139.54033578028609, 35.73756358928734, 96.87827569602781])

    geo_d = pyproj.CRS.from_epsg(4326)
    geo_c = pyproj.CRS.from_dict({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'})

    out_c = idp.geotools.convert_proj3d(
        geodetic, geo_d, geo_c, is_xyz=True
    )
    # x: array([[-3943658.715, 3363404.132, 3704651.343]])
    # y: array([[-3943658.709, 3363404.124, 3704651.307]])
    np.testing.assert_array_almost_equal(out_c, geocentric, decimal=1)

    out_d = idp.geotools.convert_proj3d(
        geocentric, geo_c, geo_d, is_xyz=False
    )
    # x: array([[139.540336,  35.737563,  96.849   ]])
    # y: array([[139.540336,  35.737564,  96.878276]])
    np.testing.assert_array_almost_equal(out_d[0:2], geodetic[0:2], decimal=5)
    np.testing.assert_array_almost_equal(out_d[2], geodetic[2], decimal=1)

############################
# test metashape functions #
############################

def test_read_chunk_zip_label_only():

    out_dict = idp.metashape.read_chunk_zip(
        str(test_data.metashape.lotus_psx)[:-9], "Lotus", chunk_id=0, return_label_only=True)

    assert list(out_dict.keys()) == ['label','enabled']

#########################
# test metashape object #
#########################
def test_class_init_metashape():
    m1 = idp.Metashape()
    assert m1.software == "metashape"

    # load with project_path
    m2 = idp.Metashape(project_path=test_data.metashape.goya_psx)
    assert m2.project_name == "goya_test"
    assert m2.label == 'Chunk 1'

    # load with project_path + chunk_id
    m2 = idp.Metashape(project_path=test_data.metashape.goya_psx, chunk_id=0)
    assert m2.project_name == "goya_test"
    ## check values
    assert m2.label == 'Chunk 1'
    assert m2.meta == {}
    assert len(m2.photos) == 259
    assert m2.photos[0].label == "DJI_0284.JPG"
    assert m2.photos[0].enabled == True
    assert m2.photos[0].path == "//172.31.12.56/pgg2020a/drone/20201029/goya/DJI_0284.JPG"
    assert m2.photos[0].rotation.shape == (3, 3)
    assert m2.photos[0].sensor.width == m2.sensors[0].width

def test_class_init_metashape_multi_folder():
    ms = idp.Metashape(project_path=test_data.metashape.multifolder_psx, chunk_id=0)

    assert ms.photos[0].label == "100MEDIA-DJI_0003"
    # mac\linux path
    assert "100MEDIA/DJI_0003" in ms.photos[0].path or \
        "100MEDIA\\DJI_0003"  # windows path

    # test run reverse calcuation
    # 1/30 size of square in the center of the full plot
    # export the chunk to aaa.ply, and read by open3d
    # 
    # In [2]: pts = o3d.io.read_point_cloud("/Users/hwang/Desktop/aaa.ply")
    # In [4]: points = np.array(pts.points)
    # In [8]: xmin, ymin, z_min = points.min(axis=0)
    # In [9]: xmax, ymax, z_max = points.max(axis=0)
    # In [10]: xmean, ymean, z_mean = points.mean(axis=0)
    # In [11]: xlen = xmax- xmin
    # In [12]: ylen = ymax-ymin
    #
    # In [13]: xlen
    # Out[13]: 0.0045318603515625
    # In [26]: bf = xlen/30
    # In [27]: x0 = xmean - bf
    # In [28]: y0 = ymean -bf
    # In [29]: x1 = xmean + bf
    # In [30]: y1 = ymean + bf
    # In [31]: roi = np.array([[x0, y0, z0],[x0, y1, z0],[x1,y1,z0],[x1,y0,z0],[x0,y0,
    #     ...: z0]])
    # In [32]: roi
    # Out[32]: 
    # array([[-120.87534231,   39.50387817, 1441.56202452],
    #        [-120.87534231,   39.50418029, 1441.56202452],
    #        [-120.87504018,   39.50418029, 1441.56202452],
    #        [-120.87504018,   39.50387817, 1441.56202452],
    #        [-120.87534231,   39.50387817, 1441.56202452]])

    roi = np.array(
        [[-120.87534231,   39.50387817, 1441.56202452],
         [-120.87534231,   39.50418029, 1441.56202452],
         [-120.87504018,   39.50418029, 1441.56202452],
         [-120.87504018,   39.50387817, 1441.56202452],
         [-120.87534231,   39.50387817, 1441.56202452]])

    out = ms.back2raw_crs(roi)

    assert len(out) == 40
    assert "100MEDIA-DJI_0099" in out.keys()

    # test_class_init_metashape_nested_folder
    ms = idp.Metashape(project_path=test_data.metashape.nestedfolder_psx, chunk_id=0)

    assert ms.photos[0].label == "[0]100MEDIA-DJI_0001"
    assert len(ms.photos) == 218

def test_class_init_metashape_warns_errors():
    # warning init with chunk_id without project_path
    with pytest.warns(UserWarning, match=re.escape("Unable to open chunk_id [0] for empty project with project_path=None")):
        m3 = idp.Metashape(chunk_id=0)

    # warning with multiple chunks:
    with pytest.warns(UserWarning, match=re.escape("The project has [4] chunks, however no chunk_id has been specified, open the first chunk [1] 'multiple_bbb' by default.")):
        m4 = idp.Metashape(project_path=test_data.metashape.multichunk_psx)

    # warning with unable for further anaylsys
    with pytest.warns(UserWarning, match=re.escape("Current chunk missing required ['transform', 'sensors', 'photos'] information (is it an empty chunk without finishing SfM tasks?) and unable to do further analysis.")):
        m4 = idp.Metashape(project_path=test_data.metashape.multichunk_psx, chunk_id=1)
        
    m4 = idp.Metashape(project_path=test_data.metashape.multichunk_psx)

    # trace errors for empty chunk:
    with pytest.raises(TypeError, match=re.escape("Unable to process disabled chunk (.enabled=False)")):
        m4.back2raw('roi')
    with pytest.raises(TypeError, match=re.escape("Unable to process disabled chunk (.enabled=False)")):
        m4.back2raw_crs('roi')
    with pytest.raises(TypeError, match=re.escape("Unable to process disabled chunk (.enabled=False)")):
        m4.get_photo_position('roi')
    with pytest.raises(TypeError, match=re.escape("Unable to process disabled chunk (.enabled=False)")):
        m4.sort_img_by_distance('img_dict_all', 'roi')

    # trace warnings when not given correct CRS
    m5 = idp.Metashape(test_data.metashape.lotus_psx)
    plot = np.ones((5,3)) * np.array([360000, 3950000, 100])

    with pytest.warns(UserWarning, match=re.escape("Have not specify the CRS of output DOM/DSM/PCD, ")):
        m5.back2raw_crs(plot)

    with pytest.warns(UserWarning, match=re.escape("Have not specify the CRS of output DOM/DSM/PCD, ")):
        roi = idp.ROI()
        roi['plot1'] = plot
        m5.back2raw(roi)


def test_class_init_metashape_with_missing_chunk_folders():     
    m6 = idp.Metashape(project_path=test_data.metashape.two_calib_psx)
    assert len(m6._chunk_id2label) == 1 
    assert '0' in m6._chunk_id2label.keys()
    assert 'RGB_230923_20m' in m6._chunk_id2label.values()
            

def test_class_fetch_by_label():
    m2 = idp.Metashape(project_path=test_data.metashape.goya_psx, chunk_id="Chunk 1")

    assert m2.label == 'Chunk 1'


def test_class_fetch_by_label_error():
    with pytest.raises(KeyError, match=re.escape("Could not find chunk_id [21] in")):
        m2 = idp.Metashape(project_path=test_data.metashape.multichunk_psx, chunk_id="21")

    with pytest.warns(UserWarning, match=re.escape("This project only has one chunk named [0] 'Chunk 1', ignore the wrong chunk_id [21] specified by user.")):
        m3 = idp.Metashape(project_path=test_data.metashape.goya_psx, chunk_id="21")

def test_class_show_chunk():
    m1 = idp.Metashape()

    assert m1._show_chunk() == "<Empty easyidp.Metashape object>"

    ms_goya = idp.Metashape(project_path=test_data.metashape.goya_psx)

    goya_show = \
        "<'goya_test.psx' easyidp.Metashape object with 1 active chunks>\n\n" \
        "  id  label\n" \
        "----  -------\n" \
        "-> 0  Chunk 1"

    assert ms_goya._show_chunk().replace(' ', '') == goya_show.replace(' ', '')

    ms_lotus = idp.Metashape(project_path=test_data.metashape.lotus_psx)

    lotus_show = \
        "<'Lotus.psx' easyidp.Metashape object with 1 active chunks>\n\n" \
        "  id  label\n" \
        "----  -------\n" \
        "-> 0  Chunk 1"

    assert ms_lotus._show_chunk().replace(' ', '') == lotus_show.replace(' ', '')

    m4 = idp.Metashape(project_path=test_data.metashape.multichunk_psx, chunk_id=1)
    m4_show = \
        "<'multichunk.psx' easyidp.Metashape object with 4 active chunks>\n\n" \
        "  id  label\n" \
        "----  --------------\n" \
        "-> 1  multiple_bbb\n" \
        "   2  multiple_aaa\n" \
        "   3  multiple_aaa_1\n" \
        "   4  multiple_aaa_2"
    
    assert m4._show_chunk().replace(' ', '') == m4_show.replace(' ', '')

    # test the ability to handle duplicated name
    assert list(m4._label2chunk_id.keys()) == list(m4._chunk_id2label.values())
    assert list(m4._label2chunk_id.values()) == list(m4._chunk_id2label.keys())

def test_class_open_chunk_print():

    m4 = idp.Metashape(project_path=test_data.metashape.multichunk_psx, chunk_id=1)

    # open chunk by id
    m4.open_chunk('4')
    m4_show_id = \
        "<'multichunk.psx' easyidp.Metashape object with 4 active chunks>\n\n" \
        "  id  label\n" \
        "----  --------------\n" \
        "   1  multiple_bbb\n" \
        "   2  multiple_aaa\n" \
        "   3  multiple_aaa_1\n" \
        "-> 4  multiple_aaa_2"
    
    assert m4._show_chunk().replace(' ', '') == m4_show_id.replace(' ', '')

    # open chunk by label
    m4.open_chunk('multiple_aaa')
    m4_show_label = \
        "<'multichunk.psx' easyidp.Metashape object with 4 active chunks>\n\n" \
        "  id  label\n" \
        "----  --------------\n" \
        "   1  multiple_bbb\n" \
        "-> 2  multiple_aaa\n" \
        "   3  multiple_aaa_1\n" \
        "   4  multiple_aaa_2"
    
    assert m4._show_chunk().replace(' ', '') == m4_show_label.replace(' ', '')

def test_metashape_show_photo_folder(capfd):
    ms = idp.Metashape(
        test_data.metashape.lotus_psx, 
        chunk_id=0, 
    )

    ms.show_photo_folder()

    out, err = capfd.readouterr()

    assert out == f"\'{str(test_data.data_dir / 'metashape' / '20170531' / 'photos')}\': [DJI_0422.JPG, DJI_0423.JPG, ..., DJI_0571.JPG, DJI_0572.JPG] (151 photos)\n\n"

def test_metashape_change_raw_img_folder():
    # test with string input
    with pytest.raises(NotADirectoryError, match=re.escape("The given folder [Path/not/exists] not exists")):
        ms = idp.Metashape(
            test_data.metashape.lotus_psx, 
            chunk_id=0, 
            raw_img_folder="Path/not/exists",
            check_img_existance=True
        )

    with pytest.raises(FileNotFoundError, match=re.escape("Could not find image file")):
        ms = idp.Metashape(
            test_data.metashape.lotus_psx, 
            chunk_id=0, 
            raw_img_folder=test_data.data_dir / "pix4d" / "lotus_tanashi_full" / "photos",
            check_img_existance=True
        )

    changed_folder = test_data.data_dir / "pix4d" / "lotus_tanashi_full" / "photos"

    ms = idp.Metashape(
        test_data.metashape.lotus_psx, 
        chunk_id=0, 
        raw_img_folder=changed_folder,
        check_img_existance=False
    )

    assert ms.photos[0].path == str(changed_folder / "DJI_0422.JPG")
    

def test_local2world2local():
    attempt1 = idp.Metashape()
    attempt1.transform.matrix = np.asarray([
        [-0.86573098, -0.01489186,  0.08977677,  7.65034123],
        [ 0.06972335,  0.44334391,  0.74589315,  1.85910928],
        [-0.05848325,  0.74899678, -0.43972184, -0.1835615],
        [ 0.,          0.,          0.,           1.]], dtype=float)
    w_pos = np.array([0.5, 1, 1.5])
    l_pos = np.array(
        [7.960064093299587, 1.3019528769064523, -2.6697181763370965]
    )
    w_pos_ans = np.array(
        [0.4999999999999978, 0.9999999999999993, 1.5]
    )

    world_pos = attempt1._local2world(l_pos)
    np.testing.assert_array_almost_equal(w_pos_ans, world_pos, decimal=6)

    local_pos = attempt1._world2local(w_pos)
    np.testing.assert_array_almost_equal(l_pos, local_pos, decimal=6)


def test_metashape_project_local_points_on_raw():
    chunk = idp.Metashape(test_data.metashape.goya_psx, chunk_id=0)

    # test for single point
    l_pos = np.array([7.960064093299587, 1.3019528769064523, -2.6697181763370965])

    p_dis_out = chunk._back2raw_one2one(
        l_pos, photo_id=0, distortion_correct=False)
    p_undis_out = chunk._back2raw_one2one(
        l_pos, photo_id=0, distortion_correct=True)

    # pro_api_out = np.asarray([2218.883386793118, 1991.4709388015149])
    my_undistort_out = np.array([2220.854889556147, 1992.6933680261686])
    my_distort_out = np.array([2218.47960556, 1992.46356322])

    np.testing.assert_array_almost_equal(p_dis_out, my_distort_out)
    np.testing.assert_array_almost_equal(p_undis_out, my_undistort_out)

    # test for multiple points
    l_pos_points = np.array([
        [7.960064093299587, 1.3019528769064523, -2.6697181763370965],
        [7.960064093299587, 1.3019528769064523, -2.6697181763370965]])

    p_dis_outs = chunk._back2raw_one2one(
        l_pos_points, photo_id=0, distortion_correct=False)
    p_undis_outs = chunk._back2raw_one2one(
        l_pos_points, photo_id=0, distortion_correct=True)

    my_undistort_outs = np.array([
        [2220.854889556147, 1992.6933680261686],
        [2220.854889556147, 1992.6933680261686]])
    my_distort_outs = np.array([
        [2218.47960556, 1992.46356322],
        [2218.47960556, 1992.46356322]])

    np.testing.assert_array_almost_equal(p_dis_outs, my_distort_outs)
    np.testing.assert_array_almost_equal(p_undis_outs, my_undistort_outs)


def test_world2crs_and_on_raw_images():
    chunk = idp.Metashape(test_data.metashape.wheat_psx, chunk_id=0)

    local = np.array(
        [11.870130675203006, 0.858098777517136, -12.987136541275])
    geocentric = np.array(
        [-3943658.7087006606, 3363404.124223561, 3704651.3067566575])
    # columns=['lon', 'lat', 'alt']
    geodetic = np.array(
        [139.54033578028609, 35.73756358928734, 96.87827569602781])

    idp_world = chunk._local2world(local)
    np.testing.assert_array_almost_equal(idp_world, geocentric, decimal=1)

    idp_crs = chunk._world2crs(idp_world)
    np.testing.assert_array_almost_equal(idp_crs, geodetic)

    camera_id = 56
    camera_label = "DJI_0057"
    camera_pix_ans = np.array([2391.7104647010146, 1481.8987733175165])

    idp_cam_pix = chunk._back2raw_one2one(local, camera_id, distortion_correct=True)
    np.testing.assert_array_almost_equal(camera_pix_ans, idp_cam_pix)

    idp_cam_pix_l = chunk._back2raw_one2one(local, camera_label, distortion_correct=True)
    np.testing.assert_array_almost_equal(camera_pix_ans, idp_cam_pix_l)


def test_class_back2raw_and_crs():
    ms = idp.Metashape(project_path=test_data.metashape.lotus_psx, chunk_id=0)

    # roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
    # # only pick 2 plots as testing data
    # key_list = list(roi.keys())
    # for key in key_list:
    #     if key not in ["N1W1", "N1W2"]:
    #         del roi[key]
    # roi.get_z_from_dsm(test_data.metashape.lotus_dsm)

    roi = roi_select.copy()
    roi.get_z_from_dsm(test_data.metashape.lotus_dsm)

    poly = roi["N1W2"]
    ms.crs = roi.crs

    out = ms.back2raw_crs(poly)

    # out_err = ms.back2raw_crs(roi_err['N1W2'])

    assert len(out) == 21

    out_all = ms.back2raw(roi)

    assert len(out_all) == 4
    assert isinstance(out_all["N1W2"], dict)


def test_class_back2raw_error():
    ms = idp.Metashape(test_data.metashape.lotus_psx)

    roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)

    with pytest.raises(
        ValueError, 
        match=re.escape(
            "The back2raw function requires 3D roi with shape=(n, 3), but [N1W1] is (5, 2)")):
            ms.back2raw(roi)


def test_metashape_get_photo_position():
    ms = idp.Metashape(test_data.metashape.lotus_psx, chunk_id=0)

    out = ms.get_photo_position()

    assert len(out) == 151
    assert "DJI_0430" in out.keys()
    np.testing.assert_almost_equal(out["DJI_0430"], np.array([139.5405732 ,  35.73445975, 128.96422715]))
    
    # convert to another proj
    out_utm = ms.get_photo_position(pyproj.CRS.from_epsg(32654), refresh=True)
    np.testing.assert_almost_equal(out_utm['DJI_0430'], np.array([ 368021.21565782, 3955478.61203427,     128.96422715]))

    # change crs and refresh
    ms.crs = pyproj.CRS.from_epsg(32654)
    assert ms._photo_position_cache is None
    out_utm = ms.get_photo_position()
    np.testing.assert_almost_equal(out_utm['DJI_0430'], np.array([ 368021.21565782, 3955478.61203427,     128.96422715]))


def test_debug_discussion_12():
    ms = idp.Metashape()

    pos = np.array([-3943837.80419438,  3363533.48603071,  3704402.1784295 ])
    ms.crs = pyproj.CRS.from_epsg(32654)

    '''
    The following code will raise the following error:

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "D:\OneDrive\Program\GitHub\EasyIDP\easyidp\metashape.py", line 99, in _world2crs
        return convert_proj3d(points_np, self.world_crs, self.crs)
      File "D:\OneDrive\Program\GitHub\EasyIDP\easyidp\metashape.py", line 1047, in convert_proj3d
        return out[0, :]
    raise UnboundLocalError: local variable 'out' referenced before assignment

    Reasons:
    is_xyz = True -> crs_target(epsg32654) neither is_geocentric nor is_geographic, it is `is_projected`

    <Derived Projected CRS: EPSG:32654>
    Name: WGS 84 / UTM zone 54N
    Axis Info [cartesian]:
    - E[east]: Easting (metre)
    - N[north]: Northing (metre)
    Area of Use:
    - name: Between 138°E and 144°E, northern hemisphere between equator and 84°N, onshore and offshore. Japan. Russian Federation.
    - bounds: (138.0, 0.0, 144.0, 84.0)
    Coordinate Operation:
    - name: UTM zone 54N
    - method: Transverse Mercator
    Datum: World Geodetic System 1984 ensemble
    - Ellipsoid: WGS 84
    - Prime Meridian: Greenwich
    '''
    #with pytest.raises(TypeError, match=re.escape("Given crs is neither `crs.is_geocentric=True` nor `crs.is_geographic`")):
    
    out = ms._world2crs(pos)
    np.testing.assert_almost_equal(out, np.array([ 368017.73174354, 3955492.19259721,     130.09433649]))

    back = ms._crs2world(out)
    np.testing.assert_almost_equal(back, pos)

    ms.crs = None

    out = ms._world2crs(pos)
    np.testing.assert_almost_equal(out, np.array([139.54053245,  35.73458169, 130.09433649]))

def test_debug_calibration_tag_error():
    # test ishii san's <calibration> out of index error (has one sensor tag without <calibration>)
    wrong_sensor = """
    <sensors next_id="2">
        <sensor id="0" label="Test_Pro (10.26mm)" type="frame">
            <resolution width="5472" height="3648"/>
            <property name="pixel_width" value="0.0024107142857142856"/>
            <property name="pixel_height" value="0.0024107142857142856"/>
            <property name="focal_length" value="10.26"/>
            <property name="layer_index" value="0"/>
            <bands>
                <band label="Red"/>
                <band label="Green"/>
                <band label="Blue"/>
            </bands>
            <data_type>uint8</data_type>
            <meta>
                <property name="Exif/BodySerialNumber" value="0K8TGAM0125288"/>
                <property name="Exif/Make" value="DJI"/>
                <property name="Exif/Model" value="Test_Pro"/>
                <property name="Exif/Software" value="10.00.11.04"/>
            </meta>
        </sensor>
        <sensor id="1" label="Test_Pro (10.26mm)" type="frame">
            <resolution width="5472" height="3648"/>
            <property name="pixel_width" value="0.0024107142857142856"/>
            <property name="pixel_height" value="0.0024107142857142856"/>
            <property name="focal_length" value="10.26"/>
            <property name="layer_index" value="0"/>
            <bands>
                <band label="Red"/>
                <band label="Green"/>
                <band label="Blue"/>
            </bands>
            <data_type>uint8</data_type>
            <calibration type="frame" class="adjusted">
                <resolution width="5472" height="3648"/>
                <f>3239.2350850187408</f>
                <cx>-17.266617222361305</cx>
                <cy>51.627079230267093</cy>
                <k1>-0.0061748904544724733</k1>
                <k2>0.0085312948846406767</k2>
                <k3>-0.0059426077261182687</k3>
                <p1>0.0038636434570336977</p1>
                <p2>0.00036559169480901107</p2>
            </calibration>
            <covariance>
                <params>f cx cy k1 k2 k3 p1 p2</params>
                <coeffs>1.3198921426619455e+02 -3.6441406803097598e-01 -8.8691870942484619e-01 -4.8035808438808746e-04 1.3837494678605588e-03 -1.4478477773082513e-03 1.6448496189504767e-04 1.9359169816499220e-05 -3.6441406803097598e-01 8.8492784732170038e-01 -1.8406731551178616e-02 4.5636954299254024e-06 -9.8824939561102722e-06 9.7826072595011059e-06 7.4748795744222678e-07 2.2077354984627524e-07 -8.8691870942484619e-01 -1.8406731551178616e-02 1.1190160084598506e+00 4.4433944809015497e-06 -1.4316737176840743e-05 1.3926778019564493e-05 -1.6128351018434475e-06 3.4438887312607803e-06 -4.8035808438808746e-04 4.5636954299254024e-06 4.4433944809015497e-06 3.2480428480148513e-09 -6.0445335360342845e-09 5.8919283699049205e-09 -5.4758330566968659e-10 -6.0991623945693416e-11 1.3837494678605588e-03 -9.8824939561102722e-06 -1.4316737176840743e-05 -6.0445335360342845e-09 1.7024280489663090e-08 -1.6861506240264254e-08 1.7186999180350537e-09 1.5819520020961357e-10 -1.4478477773082513e-03 9.7826072595011059e-06 1.3926778019564493e-05 5.8919283699049205e-09 -1.6861506240264254e-08 1.7046680052661823e-08 -1.7854037432768406e-09 -1.7316458740206863e-10 1.6448496189504767e-04 7.4748795744222678e-07 -1.6128351018434475e-06 -5.4758330566968659e-10 1.7186999180350537e-09 -1.7854037432768406e-09 7.4263922628734573e-10 2.0456928859553929e-11 1.9359169816499220e-05 2.2077354984627524e-07 3.4438887312607803e-06 -6.0991623945693416e-11 1.5819520020961357e-10 -1.7316458740206863e-10 2.0456928859553929e-11 5.5219741883705303e-10</coeffs>
            </covariance>
            <meta>
                <property name="Exif/BodySerialNumber" value="0K8TGAM0125288"/>
                <property name="Exif/Make" value="DJI"/>
                <property name="Exif/Model" value="Test_Pro"/>
                <property name="Exif/Software" value="10.00.11.04"/>
            </meta>
        </sensor>
    </sensors>
    """

    xml_tree = ET.ElementTree(ET.fromstring(wrong_sensor))
    sensors_search = xml_tree.findall("sensor")
    assert len(sensors_search) == 2

    for i, sensor_tag in enumerate(sensors_search):

        if i == 0:
            with pytest.warns(UserWarning, match=re.escape('No expected <calibration class="adjusted"> tag found in <sensor label=Test_Pro (10.26mm)> tag')):
                sensor = idp.metashape._decode_sensor_tag(sensor_tag)

            assert sensor.calibration is None
        else:
            sensor = idp.metashape._decode_sensor_tag(sensor_tag)

            assert sensor.calibration.f == 3239.2350850187408

def test_parse_sensors_with_same_name():
    dup_sensor_xml = '''
    <test>
    <sensors next_id="2">
      <sensor id="0" label="FC6540, DJI DL   35mm F2.8 LS ASPH (35mm)" type="frame">
        <resolution width="6016" height="4008"/>
        <property name="pixel_width" value="0.0040285449299403706"/>
        <property name="pixel_height" value="0.0040285449299403706"/>
        <property name="focal_length" value="35"/>
        <property name="layer_index" value="0"/>
        <bands>
          <band label="Red"/>
          <band label="Green"/>
          <band label="Blue"/>
        </bands>
        <data_type>uint8</data_type>
        <calibration type="frame" class="adjusted">
          <resolution width="6016" height="4008"/>
          <f>9342.1085191198836</f>
          <cx>-189.83553452556305</cx>
          <cy>-248.40233539070047</cy>
          <k1>0.0052975616132609543</k1>
          <k2>0.027912759943829309</k2>
          <k3>-0.12864398897487536</k3>
          <p1>-0.0024626716234921551</p1>
          <p2>-0.0024610186099128023</p2>
        </calibration>
        <covariance>
          <params>f cx cy k1 k2 k3 p1 p2</params>
          <coeffs>1.1740045342439880e+03 -3.4394700471517147e+01 -3.3180326357285384e+02 1.6849981987575206e-02 1.5741313147122529e-02 -8.5010002649401867e-02 5.8937284128415400e-04 -1.2820457371104787e-03 -3.4394700471517147e+01 5.6928870156399785e+02 1.0440571407055680e+02 -5.0203223275715834e-03 3.4033758363738821e-02 -1.5844202993434009e-01 7.1071918910600640e-03 2.9935751725901649e-03 -3.3180326357285384e+02 1.0440571407055680e+02 8.4323501491232173e+02 -3.0776560324552549e-03 1.6590670272864705e-02 -8.3429592861818072e-02 -1.4068952485630937e-03 1.2302961508974116e-02 1.6849981987575206e-02 -5.0203223275715834e-03 -3.0776560324552549e-03 2.0018503479389884e-06 -9.0504175352094234e-06 3.3351105317809357e-05 -1.1225180036431045e-07 -8.7892383954720855e-08 1.5741313147122529e-02 3.4033758363738821e-02 1.6590670272864705e-02 -9.0504175352094234e-06 1.3680837569344449e-04 -5.6344581604431023e-04 3.7526941935984332e-07 6.3368605124869975e-07 -8.5010002649401867e-02 -1.5844202993434009e-01 -8.3429592861818072e-02 3.3351105317809357e-05 -5.6344581604431023e-04 2.4232449739335538e-03 -1.7766537215773514e-06 -2.6766923915642131e-06 5.8937284128415400e-04 7.1071918910600640e-03 -1.4068952485630937e-03 -1.1225180036431045e-07 3.7526941935984332e-07 -1.7766537215773514e-06 1.9765062090723055e-07 -5.6279874436949450e-09 -1.2820457371104787e-03 2.9935751725901649e-03 1.2302961508974116e-02 -8.7892383954720855e-08 6.3368605124869975e-07 -2.6766923915642131e-06 -5.6279874436949450e-09 3.1830332460518177e-07</coeffs>
        </covariance>
        <meta>
          <property name="Exif/BodySerialNumber" value="f6bcb102921b4f05c347eb48aa3c6c4e"/>
          <property name="Exif/LensModel" value="DJI DL   35mm F2.8 LS ASPH"/>
          <property name="Exif/Make" value="DJI"/>
          <property name="Exif/Model" value="FC6540"/>
          <property name="Exif/Software" value="v01.11.2229"/>
        </meta>
      </sensor>
      <sensor id="1" label="FC6540, DJI DL   35mm F2.8 LS ASPH (35mm)" type="frame">
        <resolution width="6016" height="4008"/>
        <property name="pixel_width" value="0.0040285449299403706"/>
        <property name="pixel_height" value="0.0040285449299403706"/>
        <property name="focal_length" value="35"/>
        <property name="layer_index" value="0"/>
        <bands>
          <band label="Red"/>
          <band label="Green"/>
          <band label="Blue"/>
        </bands>
        <data_type>uint8</data_type>
        <calibration type="frame" class="adjusted">
          <resolution width="6016" height="4008"/>
          <f>9387.1765809220942</f>
          <cx>-189.15758198107463</cx>
          <cy>-265.14255491838492</cy>
          <k1>0.0076862161029297013</k1>
          <k2>-0.043809529098681556</k2>
          <k3>0.40166503757785404</k3>
          <p1>-0.0023837114019031116</p1>
          <p2>-0.0025772853340219823</p2>
        </calibration>
        <covariance>
          <params>f cx cy k1 k2 k3 p1 p2</params>
          <coeffs>1.2195330725028307e+03 -5.3735598672316883e+01 -3.7325165607216098e+02 1.1722819819355189e-02 1.0131250438547020e-01 -3.5632648588716120e-01 5.0113004722645562e-04 -2.0167635963142761e-03 -5.3735598672316883e+01 6.0260592898404343e+02 1.1057298434334751e+02 4.9153782827387818e-03 -1.2865261567558878e-01 6.6435130809814547e-01 8.2737700693599793e-03 2.9310257451372625e-03 -3.7325165607216098e+02 1.1057298434334751e+02 9.1959003457243728e+02 3.8389201797127022e-03 -1.5341597049181380e-01 9.1862221053245008e-01 -1.6363211896899016e-03 1.4543714011642174e-02 1.1722819819355189e-02 4.9153782827387818e-03 3.8389201797127022e-03 1.5738041771317424e-05 -2.1490224712750715e-04 9.3462158031549830e-04 7.3983108789068019e-09 2.0779815359742948e-08 1.0131250438547020e-01 -1.2865261567558878e-01 -1.5341597049181380e-01 -2.1490224712750715e-04 3.3906039833590069e-03 -1.5356957977758715e-02 -1.1686629422313892e-06 -2.2023046985842392e-06 -3.5632648588716120e-01 6.6435130809814547e-01 9.1862221053245008e-01 9.3462158031549830e-04 -1.5356957977758715e-02 7.2049482094581438e-02 5.5623589830091482e-06 1.3736512174935269e-05 5.0113004722645562e-04 8.2737700693599793e-03 -1.6363211896899016e-03 7.3983108789068019e-09 -1.1686629422313892e-06 5.5623589830091482e-06 2.3698285337728938e-07 -1.1939453965135837e-08 -2.0167635963142761e-03 2.9310257451372625e-03 1.4543714011642174e-02 2.0779815359742948e-08 -2.2023046985842392e-06 1.3736512174935269e-05 -1.1939453965135837e-08 3.9342177110061375e-07</coeffs>
        </covariance>
        <meta>
          <property name="Exif/BodySerialNumber" value="20160411011b4f05c347eb48aa3c6c4e"/>
          <property name="Exif/LensModel" value="DJI DL   35mm F2.8 LS ASPH"/>
          <property name="Exif/Make" value="DJI"/>
          <property name="Exif/Model" value="FC6540"/>
          <property name="Exif/Software" value="v01.11.2229"/>
        </meta>
      </sensor>
    </sensors>
    </test>
    '''
    xml_tree = ET.ElementTree(ET.fromstring(dup_sensor_xml))
    sensors = idp.Container()

    # the old error version
    debug_meta = {
        "project_folder": "project_folder", 
        "project_name"  : "project_name",
        "chunk_id": 0,
        "chunk_path": "frame_zip_file"
    }
    # for i, sensor_tag in enumerate(xml_tree.findall("sensor")):
    #     sensor = idp.metashape._decode_sensor_tag(sensor_tag, debug_meta)
    #     if sensor.calibration is not None:
    #         sensor.calibration.software = "metashape"
    #     # Raise errors for the old version
    #     if i == 1:
    #         with pytest.raises(KeyError, match=re.escape("The given item's label [FC6540, DJI DL   35mm F2.8 LS ASPH (35mm)] already exists -> ")):
    #             sensors[sensor.id] = sensor
    #     else:
    #         sensors[sensor.id] = sensor

    # the fixed version
    sensors = idp.metashape._sensorxml2object(
        xml_tree, debug_meta
    )

    assert len(sensors) == 2

    assert sensors[0].label == 'FC6540, DJI DL   35mm F2.8 LS ASPH (35mm)'
    assert sensors[1].label == 'FC6540, DJI DL   35mm F2.8 LS ASPH (35mm) [1]'

def test_metashape_disordered_image_xml():
    ms = idp.Metashape(test_data.metashape.camera_disorder_psx)

    for i in range(len(ms.photos)):
        assert ms.photos[i].id == i
    
    assert ms.photos[0].label == "80m/e-w/card01/100MEDIA/DJI_0001.JPG"

    assert ms.photos[1].label == '80m/e-w/card01/101MEDIA/DJI_0538.JPG'


def test_parse_sensor_tags_with_multiple_calibration():
    """
    <calibration type="frame" class="initial">
        <resolution width="5280" height="3956"/>
        <f>3713.29</f>
        <cx>7.0199999999999996</cx>
        <cy>-8.7200000000000006</cy>
        <k1>-0.11257523999999999</k1>
        <k2>0.014874429999999999</k2>
        <k3>-0.027064109999999999</k3>
        <p1>9.9999999999999995e-08</p1>
        <p2>-8.5719999999999999e-05</p2>
      </calibration>
      <calibration type="frame" class="adjusted">
        <resolution width="5280" height="3956"/>
        <f>4758.8543529678982</f>
        <cx>14.842273128715597</cx>
        <cy>-2.8329842300848433</cy>
        <k1>-0.15762331681570707</k1>
        <k2>-0.16432342601626651</k2>
        <k3>0.43042327353025245</k3>
        <k4>-0.49918757322065133</k4>
        <p1>-0.0012357946245061236</p1>
        <p2>-0.0011496526754087306</p2>
      </calibration>
    """
    with pytest.warns(UserWarning, match=re.escape('Detect 2 <calibration> tags in <sensor label=M3M (12.29mm)> tag, using <calibration class="adjusted">')):
        m6 = idp.Metashape(project_path=test_data.metashape.two_calib_psx)

    assert m6.sensors[0].calibration.f == 4758.8543529678982
    assert m6.sensors[0].calibration.cx == 14.842273128715597

def test_parse_multi_spectral_camera_tags():
    multi_spectral_xml = '''
    <cameras next_id="936" next_group_id="1">
        <group id="0" label="Calibration images" type="folder">
            <camera id="932" sensor_id="0" label="DJI_20230901132303_0001_MS_G" enabled="false">
                <orientation>1</orientation>
                <reference x="119.15423136699999" y="31.623286889999999" z="45.177" yaw="290.19999999999999" pitch="0.099999999999993788" roll="-0" enabled="true" rotation_enabled="false"/>
            </camera>
            <camera id="935" sensor_id="3" master_id="932" label="DJI_20230901132303_0001_MS_NIR" enabled="false">
                <orientation>1</orientation>
            </camera>
            <camera id="933" sensor_id="1" master_id="932" label="DJI_20230901132303_0001_MS_R" enabled="false">
                <orientation>1</orientation>
            </camera>
            <camera id="934" sensor_id="2" master_id="932" label="DJI_20230901132303_0001_MS_RE" enabled="false">
                <orientation>1</orientation>
            </camera>
        </group>
        <camera id="580" sensor_id="0" component_id="0" label="DJI_20230901130208_0001_MS_G">
            <transform>-0.12310895267876176 -0.99222946444830074 0.018024307225944669 -29.677549141381292 -0.97579423991461833 0.12433783620472236 0.17990470765763514 -8.9185737325389045 -0.18074785509042671 0.0045598649721814155 -0.98351889687572625 -2.5813933981741113 0 0 0 1</transform>
            <rotation_covariance>2.2379243963339504e-08 -2.4391806918735771e-09 -8.3346207902797458e-10 -2.439180691873578e-09 9.7130495805281601e-09 4.3318462943809853e-10 -8.334620790279752e-10 4.3318462943809838e-10 9.2393829603982969e-10</rotation_covariance>
            <location_covariance>1.0374823876600216e-06 1.1424223684349538e-06 -5.5892400680936419e-06 1.1424223684349538e-06 4.973670658522419e-06 -2.3456960881169798e-05 -5.5892400680936419e-06 -2.3456960881169798e-05 0.00012152005057046991</location_covariance>
            <orientation>1</orientation>
            <reference x="119.153465857" y="31.623520850999999" z="62.052" yaw="101.99999999999999" pitch="0.099999999999993788" roll="-0" enabled="true" rotation_enabled="false"/>
        </camera>
        <camera id="583" sensor_id="3" component_id="0" master_id="580" label="DJI_20230901130208_0001_MS_NIR">
            <orientation>1</orientation>
        </camera>
        <camera id="581" sensor_id="1" component_id="0" master_id="580" label="DJI_20230901130208_0001_MS_R">
            <orientation>1</orientation>
        </camera>
        <camera id="582" sensor_id="2" component_id="0" master_id="580" label="DJI_20230901130208_0001_MS_RE">
            <orientation>1</orientation>
        </camera>
    </cameras>
    '''
    ms = idp.Metashape(project_path=test_data.metashape.multi_spectral_psx, chunk_id=0)

    assert len(ms.photos) == 936
    assert ms.photos[0].label == 'DJI_20230901130734_0092_MS_G'
    assert ms.photos[0].master_id is None
    assert ms.photos[0].enabled == True
    assert ms.photos[1].master_id == 0
    assert ms.photos[1].enabled == True

    assert ms.photos[932].enabled == False
    assert ms.photos[933].enabled == False
    assert ms.photos[934].enabled == False

    assert ms.photos[935].enabled == False
    assert ms.photos[935].label == "Calibration images-DJI_20230901132303_0001_MS_NIR"


def test_multispectral_backward_projection():
    ms = idp.Metashape(project_path=test_data.metashape.multi_spectral_psx, chunk_id=0)

    roi = idp.ROI()
    roi['XS_120'] = np.array(
        [[7.04272962e+05, 3.50072277e+06, 4.22181206e+01],
         [7.04273937e+05, 3.50072256e+06, 4.22181206e+01],
         [7.04273765e+05, 3.50072176e+06, 4.22181206e+01],
         [7.04272790e+05, 3.50072197e+06, 4.22181206e+01],
         [7.04272962e+05, 3.50072277e+06, 4.22181206e+01]]
    )
    roi.crs = pyproj.CRS.from_epsg(32650)

    out_dict = ms.back2raw(roi)

    assert len(out_dict['XS_120']) == 91

    ref_pos_g = np.array(
        [[583.11737808, 969.81943293],
         [586.18480254, 860.70755771],
         [676.25815107, 863.64827397],
         [673.13453445, 972.66183241],
         [583.11737808, 969.81943293]])
    
    ref_pos_nir = np.array(
        [[628.55271509, 987.39827669],
         [631.62505771, 877.08295325],
         [722.61540256, 880.07249764],
         [719.48230069, 990.29585355],
         [628.55271509, 987.39827669]])
    
    ref_pos_r = np.array(
        [[602.64622072, 980.96274756],
         [605.71834178, 871.10029114],
         [696.34360418, 874.06535714],
         [693.21241379, 983.84204687],
         [602.64622072, 980.96274756]])

    ref_pos_re = np.array(
        [[603.19461648, 994.17375494],
         [606.25404216, 884.29288646],
         [696.88169458, 887.27163162],
         [693.76184475, 997.06009674],
         [603.19461648, 994.17375494]])

    np.testing.assert_almost_equal(out_dict['XS_120']['DJI_20230901130226_0006_MS_G'],   ref_pos_g,   7)
    np.testing.assert_almost_equal(out_dict['XS_120']['DJI_20230901130226_0006_MS_NIR'], ref_pos_nir, 7)
    np.testing.assert_almost_equal(out_dict['XS_120']['DJI_20230901130226_0006_MS_R'],   ref_pos_r,   7)
    np.testing.assert_almost_equal(out_dict['XS_120']['DJI_20230901130226_0006_MS_RE'],  ref_pos_re,  7)