import re
import pytest
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float, img_as_ubyte

import easyidp as idp

test_data = idp.data.TestData()

def test_poly2mask_type_int():
    x=[1,7,4,1]   # horizontal coord
    y=[1,2,8,1]   # vertical coord
    
    xy = np.array([x,y]).T # int type

    width = 11
    height = 10

    mask_shp = idp.cvtools.poly2mask((width, height), xy, engine='shapely')
    mask_skm = idp.cvtools.poly2mask((width, height), xy, engine='skimage')

    # the same results from skimage.polygon > 0.18.3
    wanted_shp = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)


    wanted_pil = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    np.testing.assert_equal(mask_shp, wanted_shp)
    # np.testing.assert_equal(mask_pil, wanted_pil)
    np.testing.assert_equal(mask_skm, wanted_shp)


def test_poly2mask_type_float():
    x=[1,7,4,1]   # horizontal coord
    y=[1,2,8,1]   # vertical coord
    xy = np.array([x,y], dtype=np.float16).T   # get float type

    width = 11
    height = 10

    mask_shp = idp.cvtools.poly2mask((width, height), xy, engine='shapely')
    mask_skm = idp.cvtools.poly2mask((width, height), xy, engine='skimage')

    wanted_shp = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    wanted_pil = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    wanted_skm = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    np.testing.assert_equal(mask_shp, wanted_shp)
    np.testing.assert_equal(mask_skm, wanted_skm)

def test_poly2mask_out_of_bound():
    x=[1,7,4,1]   # horizontal coord
    y=[1,2,8,1]   # vertical coord
    xy = np.array([x,y]).T   

    with pytest.raises(ValueError, match=re.escape("The polygon coords (1, 1, 7, 8) is out of mask boundary [0, 0, 6, 6]")):
        mask = idp.cvtools.poly2mask((6, 6), xy)

def test_poly2mask_wrong_type():
    xy = [[1, 2],[3, 4]]

    with pytest.raises(TypeError, match=re.escape("The `poly_coord` only accept numpy ndarray integer and float types")):
        mask = idp.cvtools.poly2mask((6, 6), xy)

    xy = np.array([1, 2, 3, 4])
    with pytest.raises(AttributeError, match=re.escape("Only nx2 ndarray are accepted")):
        mask = idp.cvtools.poly2mask((6, 6), xy)

    xy = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(AttributeError, match=re.escape("Only nx2 ndarray are accepted")):
        mask = idp.cvtools.poly2mask((6, 6), xy)

    xy = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    with pytest.raises(TypeError, match=re.escape(f"The `poly_coord` only accept numpy ndarray integer and float types")):
        mask = idp.cvtools.poly2mask((6, 6), xy)


def test_imarray_crop_2d_rgb_rgba():
    photo_path = test_data.pix4d.lotus_photos / "DJI_0174.JPG"
    roi = np.asarray([
        [2251, 1223], 
        [2270, 1270], 
        [2227, 1263], 
        [2251, 1223]])

    fig, ax = plt.subplots(2,3, figsize=(12,8))
    # -----------------------------------------------
    # 3d rgb with int type
    imarray_rgb_int = plt.imread(photo_path)
    # imarray_rgb.shape == (3456, 4608, 3)
    img_out_rgb_int, offsets_rgb = idp.cvtools.imarray_crop(imarray_rgb_int, roi, transparent_layer=3)

    ax[0, 1].imshow(img_out_rgb_int)
    ax[0, 1].set_title('rgb(int)')

    assert img_out_rgb_int.dtype == np.uint8
    np.testing.assert_almost_equal(img_out_rgb_int[0,0,:], np.asarray([145, 124, 103,   0]))

    # 3d rgb with float type
    imarray_rgb_float = img_as_float(imarray_rgb_int)
    img_out_rgb_float, offsets_rgbf = idp.cvtools.imarray_crop(imarray_rgb_float, roi, transparent_layer=3)

    ax[1, 1].imshow(img_out_rgb_float)
    ax[1, 1].set_title('rgb(float)')

    assert img_out_rgb_float.max() <= 1
    assert img_out_rgb_float.dtype == np.float64
    np.testing.assert_almost_equal(img_out_rgb_float[0,0,:], np.asarray([0.56862745, 0.48627451, 0.40392157, 0.        ]))

    # -----------------------------------------------
    # 2d
    imarray_2d_float255 = idp.cvtools.rgb2gray(imarray_rgb_int)

    im_out_2d_float255, offsets_2d = idp.cvtools.imarray_crop(imarray_2d_float255, roi)

    ax[1, 0].imshow(im_out_2d_float255, cmap='gray')
    ax[1, 0].set_title('gray(float255)')

    assert im_out_2d_float255.dtype == np.float64

    imarray_2d_int255 = imarray_2d_float255.astype(np.uint8)
    im_out_2d_int255, offsets_2d = idp.cvtools.imarray_crop(imarray_2d_int255, roi)

    ax[0, 0].imshow(im_out_2d_int255, cmap='gray')
    ax[0, 0].set_title('gray(int)')

    assert im_out_2d_int255.dtype == np.uint8
    assert im_out_2d_int255[20, 20] == 144
    # -----------------------------------------------
    # rgba
    imarray_rgba_int = np.dstack((imarray_rgb_int, np.ones((3456, 4608)) * 255)).astype(np.uint8)
    # imarray_rgba.shape == (3456, 4608, 4)

    im_out_rgba_int, offsets_rgba = idp.cvtools.imarray_crop(imarray_rgba_int, roi, transparent_layer=3)
    ax[0, 2].imshow(im_out_rgba_int)
    ax[0, 2].set_title('rgba(int)')

    assert im_out_rgba_int.dtype == np.uint8
    np.testing.assert_almost_equal(im_out_rgba_int[20, 20, :], np.asarray([163, 138, 133,  255]))

    imarray_rgba_float = img_as_float(imarray_rgba_int)

    im_out_rgba_float, offsets_rgba = idp.cvtools.imarray_crop(imarray_rgba_float, roi, transparent_layer=3)
    ax[1, 2].imshow(im_out_rgba_float)
    ax[1, 2].set_title('rgba(float)')

    assert im_out_rgba_float.dtype == np.float64
    np.testing.assert_almost_equal(im_out_rgba_float[20,20,:], np.asarray([0.63921569, 0.54117647, 0.52156863, 1.        ]))

    plt.savefig(test_data.cv.out / "imarray_clip_test.png")

    # then check the results
    expected_offsets = np.array([2227, 1223])
    np.testing.assert_equal(offsets_2d, expected_offsets)
    np.testing.assert_equal(offsets_rgb, expected_offsets)
    np.testing.assert_equal(offsets_rgba, expected_offsets)

    assert np.all(img_out_rgb_int == im_out_rgba_int)

    assert im_out_2d_float255[20,20] == 144.8887
    np.testing.assert_equal(img_out_rgb_int[20,20,:], np.array([163, 138, 133, 255], dtype=np.uint8))

def test_imarray_crop_2d_rgb_rgba_error():
    np.random.seed(0)
    imarray_init= np.random.randint(0,255,(10,10,4), dtype=np.uint8)

    str_imarray   = imarray_init.astype(str)
    img_coord = np.asarray([[0,0],[1,1],[4,4]])

    # str type error
    with pytest.raises(
        TypeError, 
        match=re.escape(
            "The `imarray` only accept numpy ndarray integer and float types"
        )
    ):
        imarray, offsets = idp.cvtools.imarray_crop(str_imarray, img_coord)

    # These test not compatative to multi-spectal GeoTiffs, deprecated
    # int_over_255  = imarray_init.astype(np.uint16)  + 255
    # int_below_0   = imarray_init.astype(np.int16)   - 128
    # float_over_1  = imarray_init.astype(np.float32) / 255 + 1
    # float_below_0 = imarray_init.astype(np.float32) / 255 - 0.5


def test_imarray_crop_handle_polygon_hv_with_float():
    # fix issue #61
    img_coord = \
    np.asarray([[7214.31561958, 3741.17729258],
                [6090.04392943, 3062.91735589],
                [6770.00648193, 1952.53149042],
                [7901.26625814, 2624.35137412],
                [7214.31561958, 3741.17729258]])

    np.random.seed(0)
    img_np = np.random.randint(low=0, high=255, size=(10000,10000,3), dtype=np.uint8)

    # the fourth layer (id=3) as new alpha layer
    imarray, offsets = idp.cvtools.imarray_crop(img_np, img_coord, transparent_layer=3)

    assert imarray.shape == (1789, 1811, 4)
    assert imarray.dtype == np.uint8
    assert offsets[0] == 6090
    assert offsets[1] == 1952
    assert offsets.dtype == np.uint32

    # test polygon_hv with error dtypes
    with pytest.raises(
        TypeError, 
        match=re.escape(
            "Only the numpy 2d array is accepted, not <class 'list'>"
        )
    ):
        img_coord = [[0, 0], [1, 0], [1, 3]]
        imarray, offsets = idp.cvtools.imarray_crop(img_np, img_coord)

    with pytest.raises(
        TypeError, 
        match=re.escape(
            "Only the numpy 2d array is accepted, not <class 'tuple'>"
        )
    ):
        img_coord = (0, 0)
        imarray, offsets = idp.cvtools.imarray_crop(img_np, img_coord)

    with pytest.raises(
        AttributeError, 
        match=re.escape(
            "Only the 2d array (xy) is accepted, expected shape like (n, 2),  not current (3, 3)"
        )
    ):
        img_coord = np.asarray([[0, 0, 1], [1, 0, 2], [1, 3, 4]])
        imarray, offsets = idp.cvtools.imarray_crop(img_np, img_coord)

    with pytest.raises(
        AttributeError, 
        match=re.escape(
            "Only the 2d array (xy) is accepted, expected shape like (n, 2),  not current (2, 3, 3)"
        )
    ):
        img_coord = np.asarray([[[0, 0, 1], [1, 0, 2], [1, 3, 4]], [[0, 0, 1], [1, 0, 2], [1, 3, 4]]])
        imarray, offsets = idp.cvtools.imarray_crop(img_np, img_coord)

    with pytest.raises(
        TypeError, 
        match=re.escape(
            "Only polygon coordinates with [np.interger] and [np.floating] are accepted, not dtype('<U"  # MacOS: <U21'), Win: <U11
        )
    ):
        img_coord = np.asarray([[0, 0], [1, 0], [1, 3]]).astype(str)
        imarray, offsets = idp.cvtools.imarray_crop(img_np, img_coord)
    


def test_roi_smaller_than_one_pixel_error():   # disucssion #39
    one_dim_imarray = np.array([255,255,255])
    polygon_hv = np.array([[1, 1], [2, 2], [1, 3], [1, 1]])
    with pytest.raises(
        ValueError, 
        match=re.escape(
            f"Current image shape {one_dim_imarray.shape} is not standard image array"
        )
    ):
        idp.cvtools.imarray_crop(one_dim_imarray, polygon_hv)