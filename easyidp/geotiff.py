import os
import pyproj
import numpy as np
import tifffile as tf
import warnings
from tqdm import tqdm
from pathlib import Path
from pyproj.exceptions import CRSError

import easyidp as idp


class GeoTiff(object):
    """A GeoTiff class, consisted by header information and file path to raw file.
    """

    def __init__(self, tif_path="", transparent_layer=None):
        """The method to initialize the GeoTiff class

        Parameters
        ----------
        tif_path : str | pathlib.Path, optional
            the path to geotiff file, by default ""
        Transparent_layer: None | int, optional
            the transparent | alpha layer of given GeoTiff file, by default None
            |    None: no alpha layer
            |    0-x : specific alpha layer
            |    -1  : the last layer

        Example
        -------

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()
            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)

        """
        self.file_path = os.path.abspath(tif_path)
        """The file path of current GeoTiff
        
        .. code-block:: python
        
            >>> dom.file_path
            PosixPath('/Users/<user>/Library/Application Support/easyidp.data/data_for_tests/pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif')
        
        """

        self.header = None
        """The Geotiff meta infomation
        
        .. code-block:: python
        
            >>> dom.header
            {'height': 5752, 'width': 5490, 'dim': 4, 'nodata': 0, 'dtype': dtype('uint8'), 
            'tags': <tifffile.TiffTags @0x00007FB1E8C3FFD0>, 'photometric': <PHOTOMETRIC.RGB: 2>, 
            'planarconfig': <PLANARCONFIG.CONTIG: 1>, 'compress': <COMPRESSION.LZW: 5>, 
            'scale': [0.00738, 0.00738], 'tie_point': [368014.54157, 3955518.2747700005], 
            'crs': <Derived Projected CRS: EPSG:32654>
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
            }
            >>> dom.header["height"]
            5752

        """

        #: The numpy ndarray of GeoTiff images
        self.imarray = None

        #: The layer to represent transparency / alpha
        self.transparent_layer = None

        if tif_path != "":
            self.read_geotiff(tif_path)

    @property
    def crs(self):
        """A quick access to ``self.header['crs']``, please access the ``header`` dict to change value"""
        if isinstance(self.header, dict) and 'crs' in self.header.keys():
            return self.header['crs']
        else:
            return None
        
    @property
    def height(self):
        """A quick access to ``self.header['height']``, please access the ``header`` dict to change value"""
        if isinstance(self.header, dict) and 'height' in self.header.keys():
            return self.header['height']
        else:
            return None
        
    @property
    def width(self):
        """A quick access to ``self.header['width']``, please access the ``header`` dict to change value"""
        if isinstance(self.header, dict) and 'width' in self.header.keys():
            return self.header['width']
        else:
            return None
        
    @property
    def dim(self):
        """A quick access to ``self.header['dim']``, please access the ``header`` dict to change value"""
        if isinstance(self.header, dict) and 'dim' in self.header.keys():
            return self.header['dim']
        else:
            return None
        
    @property
    def nodata(self):
        """A quick access to ``self.header['nodata']``, please access the ``header`` dict to change value"""
        if isinstance(self.header, dict) and 'nodata' in self.header.keys():
            return self.header['nodata']
        else:
            return None
        
    @property
    def scale(self):
        """A quick access to ``self.header['scale']``, please access the ``header`` dict to change value"""
        if isinstance(self.header, dict) and 'scale' in self.header.keys():
            return self.header['scale']
        else:
            return None
        
    @property
    def tie_point(self):
        """A quick access to ``self.header['tie_point']``, please access the ``header`` dict to change value"""
        if isinstance(self.header, dict) and 'tie_point' in self.header.keys():
            return self.header['tie_point']
        else:
            return None


    def read_geotiff(self, tif_path):
        """Open and get the meta information (header) from geotiff

        Parameters
        ----------
        tif_path : str | pathlib.Path
            the path to geotiff file

        Example
        -------
        Though this function can be used by:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()
            >>> dom = idp.GeoTiff()
            >>> dom.read_geotiff(test_data.pix4d.lotus_dom)

        It is highly recommended to specify the geotiff path when initializing the geotiff object:

        .. code-block:: python

            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)

        """
        tif_path = Path(tif_path)
        if tif_path.exists():
            self.file_path = tif_path
            self.header = get_header(str(tif_path))
        else:
            warnings.warn(f"Can not find file [{tif_path}], skip loading")


    def has_data(self):
        """Return True if current objects has geotiff infomation

        Returns
        -------
        bool

        Example
        -------

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> aaa = idp.GeoTiff()
            >>> aaa.has_data()
            False
            >>> aaa.read_geotiff(test_data.pix4d.lotus_dom)
            >>> aaa.has_data()
            True

            >>> bbb = idp.GeoTiff(test_data.pix4d.lotus_dom)
            >>> bbb.has_data()
            True
        """
        if self.header is None or not os.path.exists(self.file_path):
            return False
        else:
            return True


    def _not_empty(self):
        # check before doing functions
        if not self.has_data():
            raise FileNotFoundError("Could not operate if not specify correct geotiff file")


    def point_query(self, points_hv, is_geo=True):
        """Get the pixel value of given point(s)

        Parameters
        ----------
        points_hv : tuple | list | nx2 ndarray
            | The coordinates of qurey points, in order (horizontal, vertical)
        is_geo : bool, optional
            | The given polygon is geo coords ( ``True`` , default) or pixel coords ( ``False`` ) on imarray.

        Returns
        -------
        ndarray
            the obtained pixel value (RGB or height) 

        Example
        -------
        Prequirements

        .. code-block:: python

            >>> import easyidp as idp
            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)

        Query one point by tuple

        .. code-block:: python
        
            >>> # one point tuple
            >>> pts = (368023.004, 3955500.669)
            >>> dom.point_query(pts, is_geo=True)
            array([97.45558])

        Query one point by list

        .. code-block:: python

            >>> # one point list
            >>> pts = [368023.004, 3955500.669]
            >>> dom.point_query(pts, is_geo=True)
            array([97.45558])
        
        
        Query several points by list

        .. code-block:: python

            >>> pts = [
            ...    [368022.581, 3955501.054], 
            ...    [368024.032, 3955500.465]
            ... ]
            >>> dom.point_query(pts, is_geo=True)
            array([97.624344, 97.59617])

        Query several points by numpy

        .. code-block:: python

            >>> pts = np.array([
            ...    [368022.581, 3955501.054], 
            ...    [368024.032, 3955500.465]
            ... ])
            >>> dom.point_query(pts, is_geo=True)
            array([97.624344, 97.59617])

        See also
        --------
        easyidp.geotiff.point_query
        """
        self._not_empty()

        with tf.TiffFile(self.file_path) as tif:
            page = tif.pages[0]
            if is_geo:
                return point_query(page, points_hv, self.header)
            else:
                return point_query(page, points_hv)


    def crop_rois(self, roi, is_geo=True, save_folder=None):
        """Crop several ROIs from the geotiff by given <ROI> object with several polygons and polygon names

        Parameters
        ----------
        roi : easyidp.ROI | dict
            the <ROI> object created by easyidp.ROI(), or dictionary with multiple polygons.
            If you just need crop single polygon with ndarray coordinates, please use GeoTiff.crop_polygon() instead.
        is_geo : bool, optional
            whether the given polygon is pixel coords on imarray or geo coords (default)
        save_folder : str, optional
            the folder to save cropped images, use ROI indices as file_names, by default "", means not save.

        Returns
        -------
        dict,
            The dictionary with key=id and value=ndarray data

        Example
        -------
        Prepare data:

        .. code-block:: python
        
            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            # prepare dom geotiff
            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)

            # prepare several ROIs
            >>> roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
            >>> roi = roi[0:3]    # only use 3 for quick example
            >>> roi.change_crs(obj.crs)   # transform to the same CRS like DOM
            {0: array([[ 368017.7565143 , 3955511.08102276],
                       [ 368019.70190232, 3955511.49811902],
                       [ 368020.11263046, 3955509.54636219],
                       [ 368018.15769062, 3955509.13563382],
                       [ 368017.7565143 , 3955511.08102276]]), 
             1: array([[ 368018.20042946, 3955508.96051697],
                       [ 368020.14581791, 3955509.37761334],
                       [ 368020.55654627, 3955507.42585654],
                       [ 368018.601606  , 3955507.01512806],
                       [ 368018.20042946, 3955508.96051697]]), 
             2: array([[ 368018.64801755, 3955506.84956301],
                       [ 368020.59340644, 3955507.26665948],
                       [ 368021.00413502, 3955505.31490271],
                       [ 368019.04919431, 3955504.90417413],
                       [ 368018.64801755, 3955506.84956301]])}

        Use this function:

        .. code-block:: python

            >>> out_dict = obj.crop_rois(roi)
            {"N1W1": array[...], "N1W3": array[...], ...}

            >>> out_dict["N1W1"].shape
            (320, 319, 4)

        If you want automatically save geotiff results to specific folder:

        .. code-block:: python

            >>> tif_out_folder = "./cropped_geotiff"
            >>> os.mkdir(tif_out_folder)
            >>> out_dict = obj.crop_rois(roi, save_folder=tif_out_folder)

        """

        self._not_empty()
        
        if not isinstance(roi, (dict, idp.ROI)):
            raise TypeError(f"Only <dict> and <easyidp.ROI> with multiple polygons are accepted, not {type(roi)}. If it is 2D ndarray coordiante for just one polygon, please use `GeoTiff.crop_polygon()` instead.")

        pbar = tqdm(roi.items(), desc=f"Crop roi from geotiff [{os.path.basename(self.file_path)}]")
        out_dict = {}
        for k, polygon_hv in pbar:
            if save_folder is not None and Path(save_folder).exists():
                save_path = Path(save_folder) / (k + ".tif")
            else:
                save_path = None

            imarray = self.crop_polygon(polygon_hv, is_geo, save_path)

            out_dict[k] = imarray

        return out_dict


    def crop_polygon(self, polygon_hv, is_geo=True, save_path=None):
        """Crop a given polygon from geotiff

        Parameters
        ----------
        polygon_hv : numpy nx2 array
            (horizontal, vertical) points
        is_geo : bool, optional
            whether the given polygon is pixel coords on imarray or geo coords (default)
        save_path : str, optional
            if given, will save the cropped as \*.tif file to path, by default None

        Returns
        -------
        imarray_out
            The cropped numpy pixels imarray

        Example
        -------
        Prepare data:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            # prepare geotiff
            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)

            # prepare polygon
            >>> roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
            >>> roi = roi[0]
            >>> roi.change_crs(dom.crs)
            >>> roi
            array([[ 368017.7565143 , 3955511.08102276],
                   [ 368019.70190232, 3955511.49811902],
                   [ 368020.11263046, 3955509.54636219],
                   [ 368018.15769062, 3955509.13563382],
                   [ 368017.7565143 , 3955511.08102276]])

        Use this function:

        .. code-block:: python
            
            >>> imarray = dom.crop_polygon(roi, is_geo=True)
            >>> imarray.shape
            (320, 319, 4)

        If you want to save the previous as new GeoTiff:

        .. code-block:: python

            >>> save_tiff = "path/to/save/cropped.tif"
            >>> imarray = obj.crop_polygon(polygon_hv, is_geo=True, save_path=save_tiff)
            
        """
        self._not_empty()

        if is_geo:
            poly_px = geo2pixel(polygon_hv, self.header, return_index=True)
        else:
            if np.issubdtype(polygon_hv.dtype, np.floating):
                poly_px = np.floor(polygon_hv).astype(int)
                warnings.warn("The given pixel coordinates is not integer and is converted, if it is geo_coordinate, please specfiy `header=get_header()`")
            elif np.issubdtype(polygon_hv.dtype, np.integer):
                poly_px = polygon_hv
            else:
                raise TypeError("Only ndarray int and float dtype are acceptable for `polygon_hv`")

        # calculate the bbox of given region
        bbox_left_top = poly_px.min(axis=0)
        bbox_right_bottom = poly_px.max(axis=0)
        bbox_size = bbox_right_bottom - bbox_left_top

        # input order = horizontal, vertical 
        # horizontal[0]-> left distance, width
        # vertical[1] -> top distance, height
        # need to reverse here
        top = bbox_left_top[1]
        left = bbox_left_top[0]
        h = bbox_size[1]
        w = bbox_size[0]

        # crop by bbox from whole geotiff by tiffile_crop first (no need to load full image to memory)
        # (page, top, left, h, w):
        with tf.TiffFile(self.file_path) as tif:
            page = tif.pages[0]
            imarray_bbox = tifffile_crop(page, top, left, h, w)

        # then crop the polygon from the imarray_bbox
        poly_offseted_px = poly_px - bbox_left_top
        imarray_out, _ = idp.cvtools.imarray_crop(
            imarray_bbox, poly_offseted_px, 
            nodata_value=self.header['nodata'],
            transparent_layer=self.transparent_layer
        )

        # check if need save geotiff
        if save_path is not None:
            save_path = Path(save_path)
            if save_path.suffix  == ".tif":
                save_geotiff(self.header, imarray_out, bbox_left_top, save_path)

        return imarray_out


    def crop_rectangle(self, left, top, w, h, is_geo=True, save_path=None):
        """Extract a rectangle regeion crop from a GeoTIFF image file.

        .. code-block:: text

            (0,0)
            o--------------------------
            |           ^
            |           | top
            |           v
            | <-------> o=============o  ^
            |   left    |<---- w ---->|  |
            |           |             |  h
            |           |             |  |
            |           o=============o  v

        
        Parameters
        ----------
        top: int | float
            Coordinates of 
        left: int | float
            Coordinates of the top left corner of the desired crop.
        h: int | float
            Desired crop height.
        w: int | float
            Desired crop width.
        is_geo : bool, optional
            whether the given polygon is pixel coords on imarray or geo coords (default)
        save_path : str, optional
            if given, will save the cropped as \*.tif file to path
            
        Returns
        -------
        ndarray
            Extracted crop.

        Example
        -------

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> obj = idp.GeoTiff(test_data.pix4d.lotus_dom)
            >>> out = obj.crop_rectangle(left=434, top=918, w=320, h=321, is_geo=False)
            >>> out.shape
            (321, 320, 4)


        .. note::
            It is not recommended to use without specifying parameters like this:
            
            ``crop_rectiange(434, 918, 320, 321)``

            It is hard to know the exactly order

        .. caution::

            subfunction :func:`easyidp.geotiff.tifffile_crop` has the order ``(top, left, h, w)`` which is too heavy to change it.

        See also
        --------
        easyidp.geotiff.tifffile_crop
        """

        self._not_empty()

        
        if is_geo:
            polygon = np.array([[left, top], [left+w, top+h]])
            polygon_px = geo2pixel(polygon, self.header, return_index=True)

            left = polygon_px[0,0]
            top  = polygon_px[0,1]
            w    = polygon_px[1,0] - left
            h    = top - polygon_px[1,1]
            '''
            the geotiff coordiate y axis is upward, so need to reverse top-h
            otherwise will get an negative value.

            Not using abs() to fit the same logic with the geo2pixel() and
            to avoid potential logic error.

            Y    easyidp coord
            ^    o--------------------------> X
            |    |           ^
            |    |           | top
            |    |           v
            |    | <-------> o=============o  ^
            |    |   left    |<---- w ---->|  |
            |    |           |             |  h
            |    |           |             |  |
            |    |           o=============o  v
            |    v Y
            |
            o--------------------------------------------------> X
            Geotiff coordinate
            '''

        # check if in the boundary
        gw = self.header['width']
        gh = self.header['height']
        if left < 0 or top < 0 or left + w > gw or top + h > gh:
            raise IndexError(
                f"The given rectange [left {left}, top {top}, width {w}, height {h}] "
                f"can not fit into geotiff shape [0, 0, {gw}, {gh}]. \n"
                f"Please check is a geo coordinate or pixel coordinate and specify"
                f" `is_geo=True|False` correctly"
            )

        with tf.TiffFile(self.file_path) as tif:
            page = tif.pages[0]
            out = tifffile_crop(page, top, left, h, w)

        # check if need save geotiff
        if save_path is not None:
            save_path = Path(save_path)
            if save_path.suffix  == ".tif":
                save_geotiff(self.header, out, np.array([left, top]), save_path)

        return out


    def geo2pixel(self, polygon_hv):
        """Convert geotiff pixel coordinate (horizontal, vertical) to point cloud xyz coordinate (x, y, z)

        Parameters
        ----------
        points_hv : numpy nx2 array
            [horizontal, vertical] points

        Returns
        -------
        The ndarray pixel position of these points (horizontal, vertical)

        Example
        -------
        Prepare data:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            # prepare the roi data
            >>> roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)
            >>> roi.change_crs(dom.crs)
            >>> roi_test = roi[111]
            array([[ 368051.75902187, 3955484.68169527],
                   [ 368053.70441367, 3955485.09879908],
                   [ 368054.11515079, 3955483.14704415],
                   [ 368052.16020711, 3955482.73630818],
                   [ 368051.75902187, 3955484.68169527]])

        Use this function:

        .. code-block:: python

            >>> roi_test_pixel = dom.geo2pixel(roi_test)
            array([[5043.01515811, 4551.90714551],
                   [5306.6183839 , 4495.38901391],
                   [5362.27381938, 4759.85445164],
                   [5097.37630191, 4815.50973136],
                   [5043.01515811, 4551.90714551]])

        See also
        --------
        easyidp.geotiff.geo2pixel
        """
        return geo2pixel(polygon_hv, self.header)


    def pixel2geo(self, polygon_hv):
        """Convert geotiff pixel coordinate (horizontal, vertical) to point cloud xyz coordinate (x, y, z)

        Parameters
        ----------
        points_hv : numpy nx2 array
            [horizontal, vertical] points

        Returns
        -------
        The ndarray pixel position of these points (horizontal, vertical)

        Example
        -------
        Prepare data:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            # prepare the roi data
            >>> roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)
            >>> roi.change_crs(dom.crs)
            >>> roi_test = roi[111]
            >>> roi_test_pixel = dom.geo2pixel(roi_test)
            array([[5043.01515811, 4551.90714551],
                   [5306.6183839 , 4495.38901391],
                   [5362.27381938, 4759.85445164],
                   [5097.37630191, 4815.50973136],
                   [5043.01515811, 4551.90714551]])

        Use this function:

        .. code-block:: python

            >>> roi_test_back = dom.pixel2geo(roi_test_pixel)
            array([[ 368051.75902187, 3955484.68169527],
                   [ 368053.70441367, 3955485.09879908],
                   [ 368054.11515079, 3955483.14704415],
                   [ 368052.16020711, 3955482.73630818],
                   [ 368051.75902187, 3955484.68169527]])

        See also
        --------
        easyidp.geotiff.pixel2geo
        """

        return pixel2geo(polygon_hv, self.header)


    # def save_geotiff(self, save_path):
    #     self._not_empty()
    #     pass


    def polygon_math(self, polygon_hv, is_geo=True, kernel="mean"):
        """Calculate the valus inside given polygon

        Parameters
        ----------
        polygon_hv : numpy nx2 array | 'all'
            (horizontal, vertical) points
        is_geo : bool, optional
            whether the given polygon is pixel coords on imarray or geo coords (default)
        kernel : str, optional
            The method to calculate polygon summary, options are: ["mean", "min", "max", "pmin5", "pmin10", "pmax5", "pmax10"], please check notes section for more details.
        
        Notes
        -----
        Option details for ``kernel`` parameter:

        - "mean": the mean value inside polygon
        - "min": the minimum value inside polygon
        - "max": the maximum value inside polygon
        - "pmin5": 5th [percentile mean]_ inside polygon
        - "pmin10": 10th [percentile mean]_ inside polygon
        - "pmax5": 95th [percentile mean]_ inside polygon
        - "pmax10": 90th [percentile mean]_ inside polygon

        .. [percentile mean] the mean value of all pixels over/under xth percentile threshold
        
        Example
        -------
        Prepare data:

        .. code-block:: python
        
            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            # prepare the roi data
            >>> roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
            >>> dsm = idp.GeoTiff(test_data.pix4d.lotus_dsm)
            >>> roi.change_crs(dsm.crs)
            >>> roi_test = roi[111]
            array([[ 368051.75902187, 3955484.68169527],
                   [ 368053.70441367, 3955485.09879908],
                   [ 368054.11515079, 3955483.14704415],
                   [ 368052.16020711, 3955482.73630818],
                   [ 368051.75902187, 3955484.68169527]])

        Use this function:

        .. code-block:: python

            >>> dsm.polygon_math(roi_test, is_geo=True, kernel="mean")
            97.20491

            >>> dsm.polygon_math(roi_test, is_geo=True, kernel="pmax10")
            97.311844

        .. caution::

            This function is initially designed for doing some simple calculations one band (layer) geotiff.

            If you applying this function on RGB color geotiff, it will return the calculated results of each layer
            
            .. code-block:: python

                >>> dom.polygon_math(roi_test, is_geo=True, kernel="pmax10")
                array([139.97428808, 161.36439038, 122.30964888, 255.        ])

            The four values are RGBA four color channels.

        """
        self._not_empty()

        if isinstance(polygon_hv, str):  # == full_map
            imarray = get_imarray(self.file_path)
        else:
            imarray = self.crop_polygon(polygon_hv, is_geo)

        # remove outside values
        if len(imarray.shape) == 2:  # seems dsm
            # dim = 2
            inside_value = imarray[imarray != self.header["nodata"]]
            # all nodata
            # fix bug #69
            if len(inside_value) == 0:
                inside_value = np.array([self.header["nodata"]])
        elif len(imarray.shape) == 3 and imarray.shape[2]==4:  
            # RGBA dom
            # crop_polygon function only returns RGBA 4 layer data
            # dim = 3
            mask = imarray[:, :, 3] == 255
            inside_value = imarray[mask, :]   # (mxn-o, 4)
        else:
            raise IndexError("Only support (m,n) dsm and (m,n,4) RGBA dom")

        def _get_idx(group, thresh, compare="<="):
            if thresh.shape == ():  # single value
                if compare == "<=":
                    return group <= thresh
                else:
                    return group >= thresh
            else:
                if compare == "<=":
                    return np.all(group <= thresh, axis=1)
                else:
                    return np.all(group >= thresh, axis=1)

        if kernel == "mean":
            return np.mean(inside_value, axis=0)
        elif kernel == "min":
            return np.min(inside_value, axis=0)
        elif kernel == "max":
            return np.max(inside_value, axis=0)
        elif kernel == "pmin5":
            thresh = np.percentile(inside_value, 5, axis=0)
            idx = _get_idx(inside_value, thresh, "<=")
            return np.mean(inside_value[idx], axis=0)
        elif kernel == "pmin10":
            thresh = np.percentile(inside_value, 10, axis=0)
            idx = _get_idx(inside_value, thresh, "<=")
            return np.mean(inside_value[idx], axis=0)
        elif kernel == "pmax5":
            thresh = np.percentile(inside_value, 95, axis=0)
            idx = _get_idx(inside_value, thresh, ">=")
            return np.mean(inside_value[idx], axis=0)
        elif kernel == "pmax10":
            thresh = np.percentile(inside_value, 90, axis=0)
            idx = _get_idx(inside_value, thresh, ">=")
            return np.mean(inside_value[idx], axis=0)
        else:
            raise KeyError(f"Could not find kernel [{kernel}] in [mean, min, max, pmin5, pmin10, pmax5, pmax10]")

    # def create_grid(self, w, h, extend=False, grid_buffer=0):
    #     self._not_empty()
    #     raise NotImplementedError("This function will be provided in the future.")
        

def get_header(tif_path):
    """Read the necessary meta infomation from TIFF file

    Parameters
    ----------
    tif_path : str
        the path to the geotiff file

    Returns
    -------
    header: dict
        the container of acquired meta info

    Example
    -------
    
    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> lotus_full = idp.geotiff.get_header(test_data.pix4d.lotus_dom)
        >>> lotus_full
        {'height': 5752, 'width': 5490, 'dim': 4, 'nodata': 0, 'dtype': dtype('uint8'), 
        'tags': <tifffile.TiffTags @0x00007FD358947160>, 'photometric': <PHOTOMETRIC.RGB: 2>, 
        'planarconfig': <PLANARCONFIG.CONTIG: 1>, 'compress': <COMPRESSION.LZW: 5>, 
        'scale': [0.00738, 0.00738], 'tie_point': [368014.54157, 3955518.2747700005], 
        'crs': <Derived Projected CRS: EPSG:32654>
                Name: WGS 84 / UTM zone 54N
                Axis Info [cartesian]:
                - E[east]: Easting (metre)
                - N[north]: Northing (metre)
                Area of Use:
                - name: Between 138°E and 144°E, northern hemisphere between equator and 84°N, 
                        onshore and offshore. Japan. Russian Federation.
                - bounds: (138.0, 0.0, 144.0, 84.0)
                Coordinate Operation:
                - name: UTM zone 54N
                - method: Transverse Mercator
                Datum: World Geodetic System 1984 ensemble
                - Ellipsoid: WGS 84
                - Prime Meridian: Greenwich
        }
    """
    with tf.TiffFile(tif_path) as tif:
        header = {}
        # keys: 'width', 'height', 'dim', 'scale', 'tie_point',
        #       'nodata', 'crs', 'dtype', 'band_num'
        # for export:
        #       'tags', 'photometric', 'planarconfig', 'compression'
        page = tif.pages[0]

        header["height"] = page.shape[0]
        header["width"] = page.shape[1]
        if len(page.shape) > 2:
            # header["dim"] = page.shape[2] 
            # `band_num` used in other functions in the old version
            header["dim"] = page.samplesperpixel
        else:
            header["dim"] = 1
        header["nodata"] = page.nodata
        header["dtype"] = page.dtype

        # for save geotiff
        header["tags"] = page.tags
        header["photometric"] = page.photometric
        header["planarconfig"] = page.planarconfig
        header["compress"] = page.compression
        
        # page.geotiff_tags
        # -> 'ModelPixelScale': [0.0034900000000000005, 0.0034900000000000005, 0.0]
        header["scale"] = page.geotiff_tags["ModelPixelScale"][0:2]
        
        # page.geotiff_tags
        # -> 'ModelTiepoint': [0.0, 0.0, 0.0, 419509.89816000004, 3987344.8286, 0.0]
        header["tie_point"] = page.geotiff_tags["ModelTiepoint"][3:5]
        
        # pix4d UTM CRS:
        #    page.geotiff_tags
        #    -> 'GTCitationGeoKey': 'WGS 84 / UTM zone 54N'
        if "GTCitationGeoKey" in page.geotiff_tags.keys():
            proj_str = page.geotiff_tags["GTCitationGeoKey"]
        # metashape UTM CRS:
        #     page.geotiff_tags
        #     -> 'PCSCitationGeoKey': 'WGS 84 / UTM zone 54N'
        #     -> 'GeogCitationGeoKey': 'WGS 84'
        elif "PCSCitationGeoKey" in page.geotiff_tags.keys():
            proj_str = page.geotiff_tags["PCSCitationGeoKey"]
        else:
            # metashape: wgs 84
            #    page.geotiff_tags
            #    -> 'GeogCitationGeoKey': 'WGS 84'
            for k, v in page.geotiff_tags.items():
                if "CitationGeoKey" in k:
                    proj_str = v
                    print(f"Could not find prefered [GTCitationGeoKey, PCSCItationGeoKey], but find [{k}]={v} instead for Coordinate Reference System (CRS)")
                    break
            else:
                raise KeyError(f"Can not find Coordinate Reference System (CRS) keys '**CitationGeoKey' in Geotiff tages {page.geotiff_tags}")
        
        try:
            crs = pyproj.CRS.from_string(proj_str)
            header['crs'] = crs
        except CRSError as e:
            print(f'[io][geotiff][GeoCorrd] Generation failed, because [{e}], but you can manual specify it later by \n'
                    '>>> import pyproj \n'
                    '>>> proj = pyproj.CRS.from_epsg() # or from_string() or refer official documents:\n'
                    'https://pyproj4.github.io/pyproj/dev/api/crs/coordinate_operation.html')
            pass

    return header


def get_imarray(tif_path):
    """Read full map data as numpy array (time and RAM costy, not recommended, often requires ``4 x file_size`` of RAM)

    Parameters
    ----------
    tif_path : str
        the path to geotiff file

    Returns
    -------
    data: ndarray
        the obtained image data

    Example
    -------

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> maize_part_np = idp.geotiff.get_imarray(test_data.pix4d.maize_dom)
        >>> maize_part_np.shape
        (722, 836, 4)

    """
    with tf.TiffFile(tif_path) as tif:
        data = tif.pages[0].asarray()

    return data


def geo2pixel(points_hv, header, return_index=False):
    """Convert point cloud xyz coordinate to geotiff pixel coordinate (horizontal, vertical)

    Parameters
    ----------
    points_hv : numpy nx2 array
        [horizontal, vertical] points
    header : dict
        the geotiff head dictionary from get_header()
    return_index : bool, default false
        if false: will get float coordinates -> (23.5, 27.8)
        if true: will get int pixel index -> (23, 27)

    Returns
    -------
    ndarray 
        pixel position of these points (horizontal, vertical)

    Notes
    -----
    Please note: gis UTM coordinate, horizontal is x axis, vertical is y axis, origin at left upper.

    To crop image ndarray:

    - the first columns is vertical pixel (along height),
    - the second columns is horizontal pixel number (along width),
    - the third columns is 3 or 4 bands (RGB, alpha),
    - the x and y is reversed compared with gis coordinates.
        
    This function has already do this reverse, so that you can use the output directly.

    Example
    -------
    .. code-block:: python

        # manual specify header just as example (no need to open geotiff)
        >>> header = {'width': 19436, 'height': 31255, 'dim':4, 
                      'scale': [0.001, 0.001], 'nodata': None,
                      'tie_point': [484576.70205, 3862285.5109300003], 
                      'proj': pyproj.CRS.from_string("WGS 84 / UTM zone 53N")}
        # prepare coord data (no need to read)
        >>> gis_coord = np.asarray([
                [ 484593.67474654, 3862259.42413431],
                [ 484593.41064743, 3862259.92582402],
                [ 484593.64841806, 3862260.06515117],
                [ 484593.93077419, 3862259.55455913],
                [ 484593.67474654, 3862259.42413431]])
        # get the results
        >>> idp.geotiff.geo2pixel(gis_coord, header, return_index=True)
        array([[16972, 26086],
               [16708, 25585],
               [16946, 25445],
               [17228, 25956],
               [16972, 26086]])

    See also
    --------
    :func:`easyidp.GeoTiff.geo2pixel <easyidp.geotiff.GeoTiff.geo2pixel>`

    """
    gis_ph = points_hv[:, 0]
    gis_pv = points_hv[:, 1]

    gis_xmin = header['tie_point'][0]
    gis_ymax = header['tie_point'][1]

    scale_x = header['scale'][0]
    scale_y = header['scale'][1]

    # get float coordinate on pixels
    # - numpy_axis1 = x
    np_ax_h = (gis_ph - gis_xmin) / scale_x
    # - numpy_axis0 = y
    np_ax_v = (gis_ymax - gis_pv) / scale_y

    # get the pixel index (int)
    if return_index:  
        np_ax_h = np.floor(np_ax_h).astype(int)
        np_ax_v = np.floor(np_ax_v).astype(int)

    pixel = np.vstack([np_ax_h, np_ax_v]).T

    return pixel


def pixel2geo(points_hv, header):
    """Convert geotiff pixel coordinate (horizontal, vertical) to point cloud xyz coordinate (x, y, z)

    Parameters
    ----------
    points_hv : numpy nx2 array
        [horizontal, vertical] points
    header : dict
        the geotiff head dictionary from get_header()

    Returns
    -------
    The ndarray pixel position of these points (horizontal, vertical)

    Example
    -------
    .. code-block:: python

        >>> header = {'width': 19436, 'height': 31255, 'dim':4, 
                      'scale': [0.001, 0.001], 'nodata': None,
                      'tie_point': [484576.70205, 3862285.5109300003], 
                      'crs': pyproj.CRS.from_string("WGS 84 / UTM zone 53N")}
        >>> pixel_coord = np.asarray([
                [16972, 26086],
                [16708, 25585],
                [16946, 25445],
                [17228, 25956],
                [16972, 26086]])
        >>> idp.geotiff.pixel2geo(pix_coord, header)
        array([[16972.69654   , 26086.79569047],
               [16708.59742997, 25585.10598028],
               [16946.36805996, 25445.77883044],
               [17228.72418998, 25956.37087012],
               [16972.69654   , 26086.79569047]])
    
    See also
    --------
    :func:`easyidp.GeoTiff.pixel2geo <easyidp.geotiff.GeoTiff.pixel2geo>`
    """
    gis_ph = points_hv[:, 0]
    gis_pv = points_hv[:, 1]

    gis_xmin = header['tie_point'][0]
    gis_ymax = header['tie_point'][1]

    scale_x = header['scale'][0]
    scale_y = header['scale'][1]

    # the px is numpy axis0 (vertical, h)
    # py is numpy axis1 (horizontal, w)
    if np.issubdtype(points_hv.dtype, np.integer):
        # all integer possible means the pixel index 
        #    rather than specific coordinates
        # +0.5 to get the pixel center rather than edge
        # but in QGIS, this will cause 0.5 pixel shift
        # pix_ph = points_hv[:, 0]  # + 0.5
        # pix_pv = points_hv[:, 1]  # + 0.5
        pass
    elif np.issubdtype(points_hv.dtype, np.floating):
        # all floats possible means it is the pixel coordinates
        #    rather than pixel index
        # no need to +0.5 as image center
        # pix_ph = points_hv[:, 0]
        # pix_pv = points_hv[:, 1]
        pass
    else:
        raise TypeError(f"The `points_hv` only accept numpy ndarray integer and float types")

    pix_ph = points_hv[:, 0]
    pix_pv = points_hv[:, 1]

    gis_px = gis_xmin + pix_ph * scale_x
    gis_py = gis_ymax - pix_pv * scale_y

    gis_geo = np.vstack([gis_px, gis_py]).T

    return gis_geo


def tifffile_crop(page, top, left, h, w):  
    """Extract a crop from a TIFF image file directory (IFD) by partial loading.

    Only the tiles englobing the crop area are loaded and not the whole page.

    This is usefull for large geotiff that unableto load into RAM.

    .. code-block:: text

        (0,0)
        o--------------------------
        |           ^
        |           | top
        |           v
        | <-------> o=============o  ^
        |   left    |<---- w ---->|  |
        |           |             |  h
        |           |             |  |
        |           o=============o  v

    Parameters
    ----------
    page : TiffPage
        TIFF image file directory (IFD) from which the crop must be extracted.
    top, left: int
        Coordinates of the top left corner of the desired crop.
        top = i0 = height_st
        left = j0 = w_st
    h: int
        Desired crop height.
    w: int
        Desired crop width.
        
    Returns
    -------
    out : ndarray of shape (h, w, sampleperpixel)
        Extracted crop.

    Notes
    -----
    Modified from [1]_ , 
    
    In EasyIDP v1.0, the function is ``caas_lite.get_crop(page, i0, j0, h, w)``

    References
    ----------
    .. [1] https://gist.github.com/rfezzani/b4b8852c5a48a901c1e94e09feb34743#file-get_crop-py-L60

    Example
    -------

    .. code-block:: python

        with tf.TiffFile(maize_part_dom) as tif:
            page = tif.pages[0]

            cropped = idp.geotiff.tifffile_crop(page, top=30, left=40, h=100, w=150)

    .. caution::
        It is not recommended to use without specifying parameters like this:
        
        ``crop_rectiange(434, 918, 320, 321)``

        It is hard to know the parameter order immediately.

    See also
    --------
    :func:`easyidp.GeoTiff.crop_rectangle <easyidp.geotiff.GeoTiff.crop_rectangle>`
    """
    if page.is_tiled:
        out = _get_tiled_crop(page, top, left, h, w)
    else:
        out = _get_untiled_crop(page, top, left, h, w)

    return out


def _get_tiled_crop(page, i0, j0, h, w):
    """The submodule of self.get_crop() for those tiled geotiff

    Copied from: 
    https://gist.github.com/rfezzani/b4b8852c5a48a901c1e94e09feb34743#file-get_crop-py-L60
    """
    if not page.is_tiled:
        raise ValueError("Input page must be tiled")

    im_width = page.imagewidth
    im_height = page.imagelength
    im_pyramid = page.imagedepth
    im_dim = page.samplesperpixel

    if h < 1 or w < 1:
        raise ValueError("h and w must be strictly positive.")
        
    i1, j1 = i0 + h, j0 + w
    if i0 < 0 or j0 < 0 or i1 >= im_height or j1 >= im_width:
        raise ValueError(f"Requested crop area [({i0}, {i1}), ({j0}, {j1})] is out of image bounds ({im_height}, {im_width})")

    tile_width, tile_height = page.tilewidth, page.tilelength

    tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
    tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

    tile_per_line = int(np.ceil(im_width / tile_width))

    # older version: (img_depth, h, w, dim)
    out = np.empty((im_pyramid,
                    (tile_i1 - tile_i0) * tile_height,
                    (tile_j1 - tile_j0) * tile_width,
                    im_dim), dtype=page.dtype)

    fh = page.parent.filehandle

    for i in range(tile_i0, tile_i1):
        for j in range(tile_j0, tile_j1):
            index = int(i * tile_per_line + j)

            offset = page.dataoffsets[index]
            bytecount = page.databytecounts[index]

            fh.seek(offset)
            data = fh.read(bytecount)
            tile, indices, shape = page.decode(data, index, jpegtables=page.jpegtables)

            im_i = (i - tile_i0) * tile_height
            im_j = (j - tile_j0) * tile_width
            out[:, im_i: im_i + tile_height, im_j: im_j + tile_width, :] = tile

    im_i0 = i0 - tile_i0 * tile_height
    im_j0 = j0 - tile_j0 * tile_width

    # old version: out[:, im_i0: im_i0 + h, im_j0: im_j0 + w, :]
    return out[0, im_i0: im_i0 + h, im_j0: im_j0 + w, :]

def _get_untiled_crop(page, i0, j0, h, w):
    """The submodule of self.get_crop(), for those untiled geotiff

    Copied from: 
    https://gist.github.com/rfezzani/b4b8852c5a48a901c1e94e09feb34743#file-get_crop-py-L60
    """
    if page.is_tiled:
        raise ValueError("Input page must not be tiled")

    im_width = page.imagewidth
    im_height = page.imagelength
    im_pyramid = page.imagedepth
    im_dim = page.samplesperpixel

    if h < 1 or w < 1:
        raise ValueError("h and w must be strictly positive.")

    i1, j1 = i0 + h, j0 + w
    if i0 < 0 or j0 < 0 or i1 >= im_height or j1 >= im_width:
        raise ValueError(f"Requested crop area [({i0}, {i1}), ({j0}, {j1})] is out of image bounds ({im_height}, {im_width})")
    
    fh = page.parent.filehandle

    # -------------------------------------------------
    # for data/pix4d/maize_tanashi dom:
    # dom shape: (722, 836)
    # >>> page.dataoffsets
    # (2554, 9242, 15930, 22618,  ...)  # 6688 interval
    # >>> page.databytecounts
    # (6688, 6688, 6688, 6688,  ...)
    # >>> len(page.dataoffsets)
    # 361  # = dom.h / 2 -> read 2 row once
    # -------------------------------------------------
    # no need to calculate previous part, the geotiff tag 278
    # already have this value:
    # TiffTag 278 RowsPerStrip @94 SHORT @102 = 1
    rows_per_strip = page.tags[278].value

    # commonly, it read 1 row once,
    # no need to do extra things
    if rows_per_strip == 1: 
        read_tile_idx = np.arange(i0, i1)

    # sometime it not read once per row, it reads two rows once, etc...
    # e.g. tile.shape = (1, 2, full_width, channel_num)
    #      expected ->  (1, 1, full_width, channel_num)
    # if still just picking the first row, will
    # result in a horzontal zoom 50% result
    else:
        # recalculate the row id to read
        # >>> idx = array([10, 11, 12, 13, 14])   # orignal line index
        # >>> idx_dv = np.floor(idx / row_step).astype(int)
        # array([5, 5, 6, 6, 7])
        # >>> np.unique(idx_dv)
        # array([5, 6, 7])   -> known which index to read
        read_row_id = np.arange(i0, i1)
        read_tile_idx = np.unique(np.floor(read_row_id / rows_per_strip).astype(int))

        # Then decide the start line and end line
        # >>> idx % 2
        # array([0, 1, 0, 1, 0], dtype=int32)
        # # get start id
        st_line_id = read_row_id[0] % rows_per_strip
        # # get reversed id
        # line id
        #  3  |  |  |  |...
        #  -------------------  last read
        #  0  |  |  |  |
        #  1  |  |  |  |...
        #  2  |  |  |  |
        #  -> get the [:, 0], [:, -1] or [:, -2] index
        #     as the `ed_line_id`
        ed_line_id = read_row_id[-1] % rows_per_strip - (rows_per_strip-1)

    # crop them vertically first, then crop
    temp_out = np.empty((im_pyramid, 0, w, im_dim), dtype=page.dtype)
    for index in read_tile_idx:
        offset = page.dataoffsets[index]
        bytecount = page.databytecounts[index]

        fh.seek(offset)
        data = fh.read(bytecount)

        tile, indices, shape = page.decode(data, index, jpegtables=page.jpegtables)

        # double check if is row_step per read
        # shape -> (1, 2, full_width, channel_num)
        if shape[1] != rows_per_strip:
            raise LookupError(f"the calculated {rows_per_strip} row per read does not match {shape[1]} of tifffile decoded.")

        temp_out = np.concatenate([temp_out, tile[:, :,j0:j1,:]], axis=1)

    # return cropped result
    if rows_per_strip == 1:
        if im_dim == 1:
            # is dsm -> shape (w, h)
            return temp_out[0,:,:,0]
        else:  
            # is dom -> shape (w, h, d)         
            return temp_out[0,:,:,0:im_dim] 
    else:
        if ed_line_id == 0:
            if im_dim == 1:  
                # is dsm -> shape (w, h)
                return temp_out[0, st_line_id:, :, 0]
            else:
                # is dom -> shape (w, h, d)
                return temp_out[0, st_line_id:, :, 0:im_dim]
        else:
            if im_dim == 1:
                # is dsm -> shape (w, h)
                return temp_out[0, st_line_id:ed_line_id, :, 0]
            else:
                # is dom -> shape (w, h, d)
                return temp_out[0, st_line_id:ed_line_id, :, 0:im_dim]


def point_query(page, points_hv, header=None):
    """Get the pixel value of given point(s)

    Parameters
    ----------
    page : TiffPage
        TIFF image file directory (IFD) from which the crop must be extracted.
    points_hv : tuple | list | nx2 ndarray
        | 1. one point tuple
        |     e.g. (34.57, 45.62)
        | 2. one point list
        |     e.g. [34.57, 45.62]
        | 3. points lists
        |     e.g. [[34.57, 45.62],[35.57, 46.62]]
        | 4. 2d numpy array
        |     e.g. np.array([[34.57, 45.62],[35.57, 46.62]])
    header : dict, optional
        | the geotiff head dictionary from get_header()
        | if specified, will view the `points_hv` as geo position
        |     e.g. [longtitude, latitude]
        | if not specified, will view as pixel index
        |     e.g. [1038, 567] -> pixel id

    Returns
    -------
    values: ndarray
        the obtained pixel value (RGB or height) 

    Example
    -------

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()
        >>> header = idp.geotiff.get_header(test_data.pix4d.lotus_dsm)

        >>> point1 = (368023.004, 3955500.669)
        >>> idp.geotiff.point_query(page, point1, header)
        [97.45558]

        >>> point2 = [368023.004, 3955500.669]
        >>> idp.geotiff.point_query(page, point1, header)
        [97.45558]

        >>> points3 = [
        ...     [368022.581, 3955501.054], 
        ...     [368024.032, 3955500.465]
        ... ]
        >>> idp.geotiff.point_query(page, point3, header)
        array([97.624344, 97.59617])

        >>> point4 = np.array([
        ...     [368022.581, 3955501.054], 
        ...     [368024.032, 3955500.465]
        ... ])
        >>> idp.geotiff.point_query(page, point4, header)
        array([97.624344, 97.59617])

    See also
    --------
    :func:`easyidp.GeoTiff.point_query <easyidp.geotiff.GeoTiff.point_query>`
    """

    if isinstance(points_hv, (tuple, list, np.ndarray)):
        temp = np.array(points_hv)

        dim = len(temp.shape)
        if dim == 1 and temp.shape[0] == 2:
            # fit the one point
            points_hv = np.array([temp])
        elif dim == 2 and temp.shape[1] == 2:
            # fit the points
            points_hv = temp
        else:
            raise IndexError("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")
    else:
        raise TypeError(f"Only tuple, list, ndarray are supported, not {type(points_hv)}")

    # convert to pixel index
    if header is None:   # means point hv is pixel id
        # check if is integer
        if np.issubdtype(points_hv.dtype, np.integer):
            px = points_hv
        # if float, converted to int by floor()
        else:
            px = np.floor(points_hv).astype(int)
            warnings.warn("The given pixel coordinates is not integer and is converted, if it is geo_coordinate, please specfiy `header=get_header()`")
    else:
        px = geo2pixel(points_hv, header, return_index=True)

    # get values
    # - prepare container
    values_list = []
    for p in px:
        '''
        if dom:
            cropped.shape = (1, 1, 4)
            cropped -> array([[[0, 0, 0, rgba]]], dtype=uint8)
            wanted = cropped[0,0,:]
        if dsm:
            cropped.shape = (1, 1, 1)
            cropped -> array([[-10000.]], dtype=np.float32)
            wanted = cropped[0,0]
        '''
        cropped = tifffile_crop(page, top=p[1], left=p[0], h=1, w=1)

        if page.samplesperpixel == 1:
            values_list.append(cropped[0,0])
        else:
            values_list.append(cropped[0,0,:])

    return np.array(values_list)

def _make_empty_imarray(header, h, w, layer_num=None):
    """Produce a empty image, suit the requirement for nodata
    """
    # possible dsm with only one band
    if header["dim"] == 1:
        
        # old version: np.ones((self.img_depth, h, w, 1))
        empty_template = np.ones((h, w)) * header["nodata"]
        
    # possible RGB band
    elif header["dim"] == 3 and header["dtype"] == np.uint8:
        if layer_num == 4:
            # old version: np.ones((self.img_depth, h, w, 1))
            empty_template = np.ones((h, w, 4)).astype(np.uint8) * 255
            empty_template[:,:,3] = empty_template[:,:,3] * 0
        else:
            # old version: np.ones((self.img_depth, h, w, 1))
            empty_template = np.ones((h, w, 3)).astype(np.uint8) * 255
        
    # possible RGBA band, empty defined by alpha = 0
    elif header["dim"] == 4 and header["dtype"] == np.uint8:
        # old version: np.ones((h, w, 1))
        empty_template = np.ones((h, w, 4)).astype(np.uint8) * 255
        empty_template[:,:,3] = empty_template[:,:,3] * 0
    else:
        raise ValueError(f"Current version only support DSM, RGB and RGBA images (band expect: 1,3,4; get [{header['dim']}], dtype=np.uint8; get [{header['dtype']}])")
        
    return empty_template

def _is_empty_imarray(header, imarray):
    """Judge if current img_array is empty grids
    e.g. dsm=-10000, # (page.nodata)
            rgb=[255,255,255], [0,0,0] # pure white and black
            or RGBA with full empty alpha layer  # alpha all=0
    
    Parameters
    ----------
    img_array: np.ndarray
        the outputs of self.get_crop()
    
    Returns
    -------
    bool: True is empty image.
    
    """
    is_empty = False
    if len(imarray.shape) == 2:
        imarray_dim = 1
    elif len(imarray.shape) == 3:
        _, _, imarray_dim = imarray.shape

    if imarray_dim != header["dim"]:
        raise IndexError(f"The imarray dimention [{imarray_dim}] does not match with header dimention [{header['dim']}]")
    
    if header["dim"] == 1:
        if np.all(imarray==header["nodata"]):
            is_empty = True
    elif header["dim"] == 3:
        # for the case that use (255, 255, 255) white as background
        if np.all(imarray==255):
            is_empty = True
        # in case some use (0, 0, 0) black as background
        if np.all(imarray==0):
            is_empty = True
        # in case some header specify nodata rather than 255 and 0
        if np.all(imarray==header["nodata"]):
            is_empty = True
    # for those RGBA with alpha layers, assume alpha=0 as empty
    elif header["dim"] == 4:
        if np.all(imarray[:,:,3]==0):
            is_empty = True
    else:
        raise ValueError(f"Current version only support DSM, RGB and RGBA images (band expect: 1,3,4; get [{header['dim']}], dtype=np.uint8; get [{header['dtype']}])")

    return is_empty


def save_geotiff(header, imarray, left_top_corner, save_path):
    """Save cropped region to geotiff file

    Parameters
    ----------
    header : dict
        the geotiff head dictionary from get_header()
    imarray : ndarray
        (m, n, d) image ndarray cropped from `crop_polygon`
    left_top_corner : ndarray
        | the pixel position of image top left cornder, 
        | the order is (left, top)
    save_path : str
        the save to geotiff file path

    Example
    -------
    Prepare data:

    .. code-block:: python
    
        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> dom = idp.GeoTiff(test_data.tiff.soyweed_part)
        >>> dom_imarray = idp.geotiff.get_imarray(test_data.tiff.soyweed_part)

    If want to move each 20 pixels of this geotiff to right and bottom

    .. code-block:: python

        >>> left_top_corner = [200, 200]    # add 20 and 20 offsets
        >>> save_tiff = "path/to/save/output.tif"
        >>> idp.geotiff.save_geotiff(dom_test.header, dom_imarray, left_top_corner, save_tiff)
    
    Then using the QGIS to compare between input and output:

    .. image:: ../../_static/images/python_api/save_geotiff_offset.png
        :alt: save_geotiff_offset.png
    """
    extratags = _offset_geotiff_extratags(header, left_top_corner)

    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    # create folder if parent folder not exists
    file_ext = save_path.suffix
    parent_folder = save_path.parent

    if file_ext == ".tif":
        # create folder
        if not parent_folder.exists():
            parent_folder.mkdir()

        # write geotiff
        with tf.TiffWriter(save_path) as wtif:
            wtif.write(data=imarray, 
                       software=f"EasyIDP {idp.__version__}", 
                       photometric=header["photometric"], 
                       planarconfig=header["planarconfig"], 
                       #compression=self.header["compress"], 
                       resolution=header["scale"], 
                       extratags=extratags)
    else:
        raise TypeError(f"only *.tif file name is supported, not [{file_ext}]")
        

def _offset_geotiff_extratags(header, left_top_corner):
    """Calculate the extratags part of geotiff header when the left-top corner moved

    Parameters
    ----------
    header : dict
        the geotiff head dictionary from get_header()
    left_top_corner : ndarray
        | the pixel position of image top left cornder, 
        | the order is (left, top)

    Returns
    -------
    list
        The container for calculated extratag values
    """
    geo_corner = pixel2geo(np.array([left_top_corner]), header)
    geo_h = geo_corner[0, 0]
    geo_v = geo_corner[0, 1]

    model_tie_point = (0, 0, 0, geo_h, geo_v, 0)

    extratags = []
    for k, t in header["tags"].items():
        '''
        TiffTag 256 ImageWidth @10 SHORT @18 = 5490
        TiffTag 257 ImageLength @22 SHORT @30 = 5752
        TiffTag 258 BitsPerSample @34 SHORT[4] @230 = (8, 8, 8, 8)
        TiffTag 259 Compression @46 SHORT @54 = LZW
        TiffTag 262 PhotometricInterpretation @58 SHORT @66 = RGB
        TiffTag 273 StripOffsets @70 LONG[5752] @23246 = (46439, 46678, 46934, 47207, 4
        TiffTag 277 SamplesPerPixel @82 SHORT @90 = 4
        TiffTag 278 RowsPerStrip @94 SHORT @102 = 1
        TiffTag 279 StripByteCounts @106 LONG[5752] @238 = (239, 256, 273, 278, 296, 30
        TiffTag 284 PlanarConfiguration @118 SHORT @126 = CONTIG
        TiffTag 305 Software @130 ASCII[12] @46262 = pix4dmapper
        TiffTag 317 Predictor @142 SHORT @150 = HORIZONTAL
        TiffTag 338 ExtraSamples @154 SHORT @162 = (<EXTRASAMPLE.UNASSALPHA: 2>,)
        TiffTag 339 SampleFormat @166 SHORT[4] @46254 = ('UINT', 'UINT', 'UINT', 'UINT'
        TiffTag 33550 ModelPixelScaleTag @178 DOUBLE[3] @46274 = (0.00738, 0.00738, 0.0
        TiffTag 33922 ModelTiepointTag @190 DOUBLE[6] @46298 = (0.0, 0.0, 0.0, 368014.5
        TiffTag 34735 GeoKeyDirectoryTag @202 SHORT[32] @46346 = (1, 1, 0, 7, 1024, 0,
        TiffTag 34737 GeoAsciiParamsTag @214 ASCII[29] @46410 = WGS84 / UTM zone 54N|WG
        '''
        if k < 30000:
            # this will be automatically added by wtif.save(data=imarray) in save_geotiff function
            # the key of this step is extract "hidden" tags
            continue

        if tf.__version__ < "2020.11.26" and t.dtype[0] == '1':
            dtype = t.dtype[-1]
        else:
            dtype = t.dtype

        # <tifffile.TiffTag 33922 ModelTiepointTag @190>
        if k == 33922:
            # replace the value for this tag
            value = model_tie_point
        else:
            # other just using parent value.
            value = t.value

        extratags.append((t.code, dtype, t.count, value, True))

    return extratags