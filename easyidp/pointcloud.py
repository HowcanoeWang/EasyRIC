import os
import warnings
from datetime import datetime
from tabulate import tabulate
from pathlib import Path
from tqdm import tqdm

import numpy as np
import numpy.lib.recfunctions as rfn

import laspy
from plyfile import PlyData, PlyElement

from matplotlib.patches import Polygon

import easyidp as idp


class PointCloud(object):

    """EasyIDP defined PointCloud class, consists by point coordinates, and optionally point colors and point normals.
    """

    def __init__(self, pcd_path="", offset=[0.,0.,0.]):
        """The method to initialize the PointCloud class

        Parameters
        ----------
        pcd_path : str, optional
            The point cloud file path for loading/reading, by default "", means create an empty point cloud class
        offset : list, optional
            This parameter is used to specify your own offsets rather than the automatically calculated one.
            
            .. note::
            
                When the point cloud xyz value is too large, need to deduct duplicate values (minus offsets) to save the memory cost and increase the precision.
            
            .. caution::
            
                For some Pix4D produced pointcloud, the point cloud itself has been offseted, need manually add the offset value back.

        Example
        -------

        **Prepare**

        Cancel the numpy scientific counting method display:

        .. code-block:: python

            >>> import numpy as np
            >>> np.set_printoptions(suppress=True)

        Package loading:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

        **Read large xyz point cloud**

        Most point cloud use the CRS (GPS) coordianate as xyz values directly.

        .. code-block:: python

            >>> pcd = idp.PointCloud(test_data.pcd.maize_las)
            >>> pcd.points
            array([[ 367993.0206, 3955865.095 ,      57.9707],
                   [ 367993.146 , 3955865.3131,      57.9703],
                   [ 367992.6317, 3955867.2979,      57.9822],
                   ...,
                   [ 368014.7912, 3955879.4943,      58.0219],
                   [ 368014.1528, 3955883.5785,      58.0321],
                   [ 368016.7278, 3955874.1188,      57.9668]])
        
        If store these values directly, will cost a lot of memeory with precision loss. But with offsets, the data can be stored more neatly in the EasyIDP:

        .. code-block:: python

            >>> pcd.offset
            array([ 367900., 3955800.,       0.])
            >>> pcd._points
            array([[ 93.0206,  65.095 ,  57.9707],
                   [ 93.146 ,  65.3131,  57.9703],
                   [ 92.6317,  67.2979,  57.9822],
                   ...,
                   [114.7912,  79.4943,  58.0219],
                   [114.1528,  83.5785,  58.0321],
                   [116.7278,  74.1188,  57.9668]])

        **Manually specify offset**

        The previous offset is calculated automatically by EasyIDP, you can also manually specify the offset values:

        .. code-block:: python

            >>> pcd = idp.PointCloud(test_data.pcd.maize_las, offset=[367800, 3955700, 50])
            >>> pcd.offset
            array([ 367800., 3955700.,       50.])
            >>> pcd._points
            array([[193.0206, 165.095 ,   7.9707],
                   [193.146 , 165.3131,   7.9703],
                   [192.6317, 167.2979,   7.9822],
                   ...,
                   [214.7912, 179.4943,   8.0219],
                   [214.1528, 183.5785,   8.0321],
                   [216.7278, 174.1188,   7.9668]])

        Though the inner stored values changed, it does not affect the final point valus:

        .. code-block:: python
        
            >>> pcd.points
            array([[ 367993.0206, 3955865.095 ,      57.9707],
                   [ 367993.146 , 3955865.3131,      57.9703],
                   [ 367992.6317, 3955867.2979,      57.9822],
                   ...,
                   [ 368014.7912, 3955879.4943,      58.0219],
                   [ 368014.1528, 3955883.5785,      58.0321],
                   [ 368016.7278, 3955874.1188,      57.9668]])

        **Read Pix4D offseted point cloud and add offset back**

        If you read the Pix4D produced point cloud directly:

        .. code-block:: python

            >>> pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
            >>> pcd
                         x        y        z  r    g    b        nx      ny      nz
                0  -18.908  -15.778   -0.779  123  103  79   nodata  nodata  nodata
                1  -18.908  -15.777   -0.78   124  104  81   nodata  nodata  nodata
                2  -18.907  -15.775   -0.802  123  103  80   nodata  nodata  nodata
              ...  ...      ...      ...      ...  ...  ...     ...     ...     ...
            42451  -15.789  -17.961   -0.847  116  98   80   nodata  nodata  nodata
            42452  -15.789  -17.939   -0.84   113  95   76   nodata  nodata  nodata
            42453  -15.786  -17.937   -0.833  115  97   78   nodata  nodata  nodata

        Here the xyz seems not the correct one, when we check the Pix4D project ``{name}_offset.xyz`` file in the param folders, we can find the offset values stored by Pix4D.

        .. code-block:: python

            >>> with open(test_data.pix4d.lotus_param / "hasu_tanashi_20170525_Ins1RGB_30m_offset.xyz", 'r') as f:
            ...     f.readlines()
            ['368043.000 3955495.000 98.000']

        This often requires user manually add that offset back to point cloud. But EasyIDP supports dealing with such situation easily:

        .. code-block:: python

            >>> p4d_offset_np = np.array([368043, 3955495,  98])
            >>> pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin, p4d_offset_np)
            >>> pcd
                            x            y        z  r    g    b        nx      ny      nz
                0  368024.092  3955479.222   97.221  123  103  79   nodata  nodata  nodata
                1  368024.092  3955479.223   97.22   124  104  81   nodata  nodata  nodata
                2  368024.093  3955479.225   97.198  123  103  80   nodata  nodata  nodata
              ...     ...          ...      ...      ...  ...  ...     ...     ...     ...
            42451  368027.211  3955477.039   97.153  116  98   80   nodata  nodata  nodata
            42452  368027.211  3955477.061   97.16   113  95   76   nodata  nodata  nodata
            42453  368027.214  3955477.063   97.167  115  97   78   nodata  nodata  nodata

        .. note::

            You can also obtain the ``p4d_offset_np`` by :class:`easyidp.Pix4D <easyidp.pix4d.Pix4D>` object:

            .. code-block:: python

                >>> p4d = idp.Pix4D(project_path   = test_data.pix4d.lotus_folder, 
                ...                 raw_img_folder = test_data.pix4d.lotus_photos, 
                ...                 param_folder   = test_data.pix4d.lotus_param))
                >>> p4d.offset_np
                array([ 368043., 3955495.,      98.])

            And feed it to the previous function:

            .. code-block:: python

                >>> pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin, p4d.offset_np)
        """
        #: the file path to the current point cloud file
        self.file_path = pcd_path
        #: the file extension to the current point cloud file
        self.file_ext = ".ply"

        self._points = None   # internal points with offsets to save memory
        #: The color (RGB) values of point cloud
        self.colors = None
        #: The normal vector values of point cloud
        self.normals = None
        #: The size of point cloud (xyz)
        self.shape = (0,3)

        self.offset = self._offset_type_check(offset)
        # BeatTiFul print strings for calling print() function
        self._btf_print = '<Empty easyidp.PointCloud object>'

        if pcd_path != "":
            self.read_point_cloud(pcd_path)

    def __str__(self) -> str:
        return self._btf_print

    def __repr__(self) -> str:
        return self._btf_print

    def __len__(self) -> int:
        return self.points.shape[0]

    def _update_btf_print(self):
        """Print Point Cloud in "DataFrame" beautiful way
        >>> print(pcd)
             x    y    z  r       g       b           nx      ny      nz
        0    1    2    3  nodata  nodata  nodata  nodata  nodata  nodata
        1    4    5    6  nodata  nodata  nodata  nodata  nodata  nodata
        2    7    8    9  nodata  nodata  nodata  nodata  nodata  nodata
        """
        head = ['','x','y','z','r','g','b','nx','ny','nz']
        data = []
        col_align = ["right"] + ["decimal"]*3 + ['left']*3 + ["decimal"]*3 

        if self.shape[0] > 6:
            show_idx = [0, 1, 2, -3, -2, -1]
        else:
            show_idx = list(range(self.shape[0]))

        for i in show_idx:
            if self.has_points():
                xyz = np.around(self.points[i,:], decimals=3).tolist()
            else:
                xyz = ['nodata'] * 3

            if self.has_colors():
                rgb = self.colors[i,:].tolist()
            else:
                rgb = ['nodata'] * 3

            if self.has_normals():
                nxyz = self.normals[i,:].tolist()
            else:
                nxyz = ['nodata'] * 3
            
            if i >= 0:
                data.append([i] + xyz + rgb + nxyz)
            if i < 0:
                data.append([self.shape[0] + i] + xyz + rgb + nxyz)

        if self.shape[0] > 6:
            data.insert(3, ['...'] * 10)
            
        self._btf_print = tabulate(data, headers=head, tablefmt='plain', colalign=col_align)

    @property
    def points(self):
        """The xyz values of point cloud
        """
        if self._points is None:
            return None
        else:
            return self._points + self._offset

    @points.setter
    def points(self, p):
        if not isinstance(p, np.ndarray):
            raise TypeError(f"Only numpy ndarray object are acceptable for setting values")
        elif self.shape != p.shape and self.shape != (0,3):
            raise IndexError(f"The given shape [{p.shape}] does not match current point cloud shape [{self.shape}]")
        else:
            self._points = p - self._offset
            self.shape = p.shape
            self._update_btf_print()

    @property
    def offset(self):
        """The offset value of point cloud

        .. caution::

            If change this value directly, the xyz value of point cloud will also be changed, just like moving the whole point cloud.

        Example
        -------
        For example, the point cloud like:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> pts = idp.PointCloud(test_data.pcd.maize_las)
            >>> pts
                            x            y        z  r    g    b  
                0  367993.021  3955865.095   57.971  28   21   17 
                1  367993.146  3955865.313   57.97   28   23   19 
                2  367992.632  3955867.298   57.982  29   22   18 
                ...     ...          ...      ...      ...  ...  ...
            49655  368014.791  3955879.494   58.022  33   28   25 
            49656  368014.153  3955883.578   58.032  30   40   26 
            49657  368016.728  3955874.119   57.967  25   20   18 
            >>> pts.offset
            array([ 367900., 3955800.,       0.])

        Change the offset directly:

        .. code-block:: python

            >>> pts.offset = [300, 200, 50]
            >>> pts
                            x        y        z  r    g    b  
                0  393.021  265.095  107.971  28   21   17 
                1  393.146  265.313  107.97   28   23   19 
                2  392.632  267.298  107.982  29   22   18 
                ...  ...      ...      ...      ...  ...  ...
            49655  414.791  279.494  108.022  33   28   25 
            49656  414.153  283.578  108.032  30   40   26 
            49657  416.728  274.119  107.967  25   20   18 

        .. caution::
        
            If you want to change the offset without affecting the point xyz values, please use :func:`update_offset_value`
        
        """
        return self._offset

    @offset.setter
    def offset(self, o):
        # the point values will change:
        # --------------------------------
        # points =  _point + offset
        #   |         |         |
        # change   no change  change
        o = self._offset_type_check(o)
        self._offset = o
        if self._points is not None:
            self._update_btf_print()

    def update_offset_value(self, off_val):
        """Change the offset value without affecting the xyz point values.

        Parameters
        ----------
        off_val : list | tuple | ndarray
            The offset values want to set

        Example
        -------
        For example, the point cloud like:
        
        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> pts = idp.PointCloud(test_data.pcd.maize_las)
            >>> pts
                            x            y        z  r    g    b  
                0  367993.021  3955865.095   57.971  28   21   17 
                1  367993.146  3955865.313   57.97   28   23   19 
                2  367992.632  3955867.298   57.982  29   22   18 
                ...     ...          ...      ...      ...  ...  ...
            49655  368014.791  3955879.494   58.022  33   28   25 
            49656  368014.153  3955883.578   58.032  30   40   26 
            49657  368016.728  3955874.119   57.967  25   20   18 
            >>> pts.offset
            array([ 367900., 3955800.,       0.])

        Change the offset without affecting the xyz values:

        .. code-block:: python

            >>> pts.update_offset_value([360000, 3955000, 50])

            >>> pts.offset
            array([ 360000., 3955000.,      50.])

            >>> pts.points
                            x            y        z  r    g    b                        nx                     ny                    nz
                0  367993.021  3955865.095   57.971  28   21   17    -0.031496062992125984    0.36220472440944884    0.9291338582677166
                1  367993.146  3955865.313   57.97   28   23   19     0.08661417322834646     0.07086614173228346    0.9921259842519685
                2  367992.632  3955867.298   57.982  29   22   18    -0.007874015748031496    0.26771653543307083    0.9606299212598425
              ...     ...          ...      ...      ...  ...  ...  ...                     ...                    ...
            49655  368014.791  3955879.494   58.022  33   28   25     0.44881889763779526    -0.14960629921259844    0.8740157480314961
            49656  368014.153  3955883.578   58.032  30   40   26     0.44881889763779526    -0.29133858267716534    0.8346456692913385
            49657  368016.728  3955874.119   57.967  25   20   18     0.3228346456692913      0.26771653543307083    0.8976377952755905

        .. caution::

            If you want to change the offset like moving point cloud (also change the xyz values), please use :func:`offset`
        """
        # the point values not change
        # --------------------------------
        # points =  _point + offset
        #   |         |         |
        # no change   change-   change+
        off_val = self._offset_type_check(off_val)
        
        if self._points is not None:
            self._points = self._points + self._offset - off_val
            self._offset = off_val
            self._update_btf_print()
        else:
            self._offset = off_val

    @staticmethod
    def _offset_type_check(off_val):
        if len(off_val) == 3:
            if isinstance(off_val, (list, tuple)):
                return np.asarray(off_val, dtype=np.float64)
            elif isinstance(off_val, np.ndarray):
                return off_val.astype(np.float64)
            else:
                raise ValueError(f"Only [x, y, z] list or np.array([x, y, z]) are acceptable, not {type(off_val)} type")
        else:
            raise ValueError(f"Please give correct 3D coordinate [x, y, z], only {len(off_val)} was given")

    def has_colors(self):
        """Returns True if the point cloud contains point colors.

        Returns
        -------
        bool
        """
        if self.colors is None:
            return False
        else:
            return True

    def has_points(self):
        """Returns True if the point cloud contains points.

        Returns
        -------
        bool
        """
        if self._points is None:
            return False
        else:
            return True

    def has_normals(self):
        """Returns True if the point cloud contains point normals.

        Returns
        -------
        bool
        """
        if self.normals is None:
            return False
        else:
            return True

    def clear(self):
        """Delete all points and make an empty point cloud
        """
        self._points = None   # internal points with offsets to save memory
        self.colors = None
        self.normals = None
        self.shape = (0,3)

        self.offset = np.array([0.,0.,0.])

        self.file_ext = ".ply"
        self.file_path = ""

    def read_point_cloud(self, pcd_path):
        """Open a new point cloud file to overwritting current file, support ply, laz, and las.

        .. caution::

            This operation will totally clear all data of current point cloud, and reopen a new point cloud.

        Parameters
        ----------
        pcd_path : str | pathlib.Path
            the path to point cloud file want to open

        Example
        -------

        Data prepare:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> aaa = idp.PointCloud(test_data.pcd.maize_las)
            >>> aaa
                            x            y        z  r    g    b                        nx                     ny                    nz
                0  367993.021  3955865.095   57.971  28   21   17    -0.031496062992125984    0.36220472440944884    0.9291338582677166
                1  367993.146  3955865.313   57.97   28   23   19     0.08661417322834646     0.07086614173228346    0.9921259842519685
                2  367992.632  3955867.298   57.982  29   22   18    -0.007874015748031496    0.26771653543307083    0.9606299212598425
            ...     ...          ...      ...      ...  ...  ...  ...                     ...                    ...
            49655  368014.791  3955879.494   58.022  33   28   25     0.44881889763779526    -0.14960629921259844    0.8740157480314961
            49656  368014.153  3955883.578   58.032  30   40   26     0.44881889763779526    -0.29133858267716534    0.8346456692913385
            49657  368016.728  3955874.119   57.967  25   20   18     0.3228346456692913      0.26771653543307083    0.8976377952755905

        This operation will totally overwrite the previous point cloud

        .. code-block:: python

            >>> aaa.read_point_cloud(test_data.pcd.lotus_las)
            >>> aaa
                        x        y        z  r    g    b        nx      ny      nz
                0  -18.908  -15.778   -0.779  123  103  79   nodata  nodata  nodata
                1  -18.908  -15.777   -0.78   124  104  81   nodata  nodata  nodata
                2  -18.907  -15.775   -0.802  123  103  80   nodata  nodata  nodata
            ...  ...      ...      ...      ...  ...  ...     ...     ...     ...
            42451  -15.789  -17.961   -0.847  116  98   80   nodata  nodata  nodata
            42452  -15.789  -17.939   -0.84   113  95   76   nodata  nodata  nodata
            42453  -15.786  -17.937   -0.833  115  97   78   nodata  nodata  nodata

        """
        if not os.path.exists(pcd_path):
            warnings.warn(f"Can not find file [{pcd_path}], skip loading")
            return

        if Path(pcd_path).suffix == ".ply":
            pts, cls, nms = read_ply(pcd_path)
        elif Path(pcd_path).suffix in [".laz", ".las"]:
            pts, cls, nms = read_laz(pcd_path)
        else:
            raise IOError("Only support point cloud file format ['*.ply', '*.laz', '*.las']")

        if self.has_points():
            self.clear()

        self.file_ext = os.path.splitext(pcd_path)[-1]
        self.file_path = os.path.abspath(pcd_path)

        if abs(np.max(pts)) > 65536:   # need offseting
            if not np.any(self._offset):    # not given any offset (0,0,0) -> calculate offset
                self._offset = np.floor(pts.min(axis=0) / 100) * 100
            self._points = pts - self._offset
        else:
            self._points = pts

        self.colors = cls
        self.normals = nms
        self.shape = pts.shape

        self._update_btf_print()

    def save(self, pcd_path):
        """Save current point cloud to a file, support ply, las, laz format.

        Parameters
        ----------
        pcd_path : str
            The file path of saved point cloud, if file extention not given, will use parent point cloud file extention.

        See also
        --------
        write_point_cloud

        """

        return self.write_point_cloud(pcd_path)

    def write_point_cloud(self, pcd_path):
        """Save current point cloud to a file, support ply, las, laz format.

        Parameters
        ----------
        pcd_path : str
            The file path of saved point cloud, if file extention not given, will use parent point cloud file extention.

        Example
        -------
        Data prepare:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
            >>> polygon = np.array([
            ...     [-18.42576599, -16.10819054],
            ...     [-18.00066757, -18.05295944],
            ...     [-16.05021095, -17.63488388],
            ...     [-16.46848488, -15.66774559],
            ...     [-18.42576599, -16.10819054]])

            >>> cropped = pcd.crop_point_cloud(polygon)
            >>> cropped
                         x        y        z  r       g       b           nx      ny      nz
                0  -18.417  -16.119   -0.641  nodata  nodata  nodata  nodata  nodata  nodata
                1  -18.409  -16.169   -0.647  nodata  nodata  nodata  nodata  nodata  nodata
                2  -18.404  -16.142   -0.634  nodata  nodata  nodata  nodata  nodata  nodata
              ...  ...      ...      ...      ...     ...     ...        ...     ...     ...
            22419  -16.065  -17.607   -0.699  nodata  nodata  nodata  nodata  nodata  nodata
            22420  -16.062  -17.578   -0.678  nodata  nodata  nodata  nodata  nodata  nodata
            22421  -16.051  -17.632   -0.692  nodata  nodata  nodata  nodata  nodata  nodata

        Use this function:

        .. code-block:: python

            >>> cropped.write_point_cloud(r"path/to/save/pointcloud.ply")

        """
        # if ply -> self.points + self.offsets
        # if las -> self.points & offset = self.offsets
        pcd_path = Path(pcd_path)
        file_ext = pcd_path.suffix

        if file_ext == "":
            warnings.warn(f"It seems file name [{pcd_path}] has no file suffix, using default suffix [{self.file_ext}] instead")
            out_path = pcd_path.with_name(f"{pcd_path.name}{self.file_ext}")
        else:
            if file_ext not in ['.ply', '.las', '.laz']:
                raise IOError("Only support point cloud file format ['*.ply', '*.laz', '*.las']")

            out_path = pcd_path

        if file_ext == ".ply":
            write_ply(
                points=self._points + self._offset, 
                colors=self.colors, 
                ply_path=out_path, 
                normals=self.normals)
        else:
            write_laz(
                points=self._points + self._offset, 
                colors=self.colors, 
                laz_path=out_path, 
                normals=self.normals, 
                offset=self._offset)

    def crop_rois(self, roi, save_folder=None):
        """Crop several ROIs by given <ROI> or dict object with several polygons and polygon names, along z-axis

        Parameters
        ----------
        roi : easyidp.ROI | dict
            | the <ROI> object created by easyidp.ROI()
            | or dict object with key as roi name and value as coordinates
        is_geo : bool, optional
            whether the given polygon is pixel coords on imarray or geo coords (default)
        save_folder : str, optional
            the folder to save cropped images, use ROI indices as file_names, by default None, means not save.

        Returns
        -------
        dict, The dictionary with key as roi name and value as <idp.PointCloud> object

        Example
        -------

        Data prepare:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
            >>> roi = roi[0:3]
            >>> header = idp.geotiff.get_header(test_data.pix4d.lotus_dom)
            >>> roi.change_crs(header['crs'])
            <easyidp.ROI> with 3 items
            [0]     N1W1
            array([[ 368017.7565143 , 3955511.08102276],
                   [ 368019.70190232, 3955511.49811902],
                   [ 368020.11263046, 3955509.54636219],
                   [ 368018.15769062, 3955509.13563382],
                   [ 368017.7565143 , 3955511.08102276]])
            [1]     N1W2
            array([[ 368018.20042946, 3955508.96051697],
                   [ 368020.14581791, 3955509.37761334],
                   [ 368020.55654627, 3955507.42585654],
                   [ 368018.601606  , 3955507.01512806],
                   [ 368018.20042946, 3955508.96051697]])
            [2]     N1W3
            array([[ 368018.64801755, 3955506.84956301],
                   [ 368020.59340644, 3955507.26665948],
                   [ 368021.00413502, 3955505.31490271],
                   [ 368019.04919431, 3955504.90417413],
                   [ 368018.64801755, 3955506.84956301]])

            >>> pcd = idp.PointCloud(test_data.pix4d.lotus_pcd, offset=[368043, 3955495,  98])
                              x            y        z  r    g    b        nx      ny      nz
                  0  368014.849  3955511.333   97.215  51   55   33   nodata  nodata  nodata
                  1  368014.853  3955511.352   97.239  49   52   33   nodata  nodata  nodata
                  2  368014.865  3955511.485   97.402  46   50   30   nodata  nodata  nodata
                ...     ...          ...      ...      ...  ...  ...     ...     ...     ...
            6235710  368055.047  3955482.412   96.997  173  168  170  nodata  nodata  nodata
            6235711  368055.051  3955482.407   97.051  154  140  129  nodata  nodata  nodata
            6235712  368055.058  3955482.373   97.107  114  93   72   nodata  nodata  nodata

        Run this function:

        .. code-block:: python

            >>> out = pcd.crop_rois(roi)
            >>> out
            {'N1W1': <easyidp.PointCloud class>, 'N1W2': <easyidp.PointCloud class>, 'N1W3': <easyidp.PointCloud class>}

        You can also save the output point cloud to given folder by:

        .. code-block:: python

            >>> out = pcd.crop_rois(roi, save_folder=r'path/to/save/folder/')
        
        See also
        --------
        crop_point_cloud
        """
        if not self.has_points():
            raise ValueError("Could not operate when PointCloud has no points")

        if not isinstance(roi, (dict, idp.ROI)):
            raise TypeError(f"Only <dict> and <easyidp.ROI> are accepted, not {type(roi)}")

        out_dict = {}
        pbar = tqdm(roi.items(), desc=f"Crop roi from point cloud [{os.path.basename(self.file_path)}]")
        for k, polygon_hv in pbar:
            if save_folder is not None and Path(save_folder).exists():
                save_path = Path(save_folder) / (k + self.file_ext)
            else:
                save_path = None

            out_dict[k] = self.crop_point_cloud(polygon_hv[:, 0:2])

            if save_path is not None:
                out_dict[k].write_point_cloud(save_path)

        return out_dict

    def crop_point_cloud(self, polygon_xy):
        """crop the point cloud along z axis

        Parameters
        ----------
        polygon_xy : nx2 ndarray
            the polygon xy coords

        Returns
        -------
        <easyidp.PointCloud> object, The cropped point cloud

        Example
        -------
        
        Data prepare:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
            >>> polygon = np.array([
            ...     [-18.42576599, -16.10819054],
            ...     [-18.00066757, -18.05295944],
            ...     [-16.05021095, -17.63488388],
            ...     [-16.46848488, -15.66774559],
            ...     [-18.42576599, -16.10819054]])

        Run this function:

        .. code-block:: python

            >>> cropped = pcd.crop_point_cloud(polygon)
            >>> cropped
                         x        y        z  r       g       b           nx      ny      nz
                0  -18.417  -16.119   -0.641  nodata  nodata  nodata  nodata  nodata  nodata
                1  -18.409  -16.169   -0.647  nodata  nodata  nodata  nodata  nodata  nodata
                2  -18.404  -16.142   -0.634  nodata  nodata  nodata  nodata  nodata  nodata
              ...  ...      ...      ...      ...     ...     ...        ...     ...     ...
            22419  -16.065  -17.607   -0.699  nodata  nodata  nodata  nodata  nodata  nodata
            22420  -16.062  -17.578   -0.678  nodata  nodata  nodata  nodata  nodata  nodata
            22421  -16.051  -17.632   -0.692  nodata  nodata  nodata  nodata  nodata  nodata

        """

        # judge whether proper data type
        if not isinstance(polygon_xy, np.ndarray):
            raise TypeError(f"Only numpy ndarray are supported as `polygon_xy` inputs, not {type(polygon_xy)}")

        # judge whether proper shape is (N, 2)
        if len(polygon_xy.shape) != 2 or polygon_xy.shape[1] != 2:
            raise IndexError(f"Please only spcify shape like (N, 2), not {polygon_xy.shape}")

        
        # calculate the bbox of polygon
        xmin, ymin = polygon_xy.min(axis=0)
        xmax, ymax = polygon_xy.max(axis=0)

        # get the xy value if point cloud
        x = self.points[:, 0]
        y = self.points[:, 1]

        # get the row (points id) that in bbox
        inbbox_bool = (x >= xmin) * (x <= xmax) * (y >= ymin) * (y <= ymax)
        # -> array([False, False, False, ..., False, False, False])
        inbbox_idx =  np.where(inbbox_bool)[0]
        # -> array([ 3394,  3395,  3396, ..., 41371, 41372, 41373], dtype=int64)

        # filter out in bbox points
        inbbox_pts = self.points[inbbox_idx, 0:2]

        # judge by matplotlib.polygon
        plt_poly = Polygon(polygon_xy)
        in_poly_idx = plt_poly.contains_points(inbbox_pts)
        # -> array([ True, False, False, ..., False, False, False])

        # pick selected row:
        pick_idx = inbbox_idx[in_poly_idx]
        # -> array([ 3394,  3503,  3614, ..., 41265, 41285, 41369], dtype=int64)

        # check whether have data
        if len(pick_idx) > 0:
            # create new Point Cloud object
            crop_pcd = PointCloud()
            crop_pcd.points = self.points[pick_idx, :]
            crop_pcd.update_offset_value(self.offset)
            if self.has_colors():
                crop_pcd.colors = self.colors[pick_idx, :]
            if self.has_normals():
                crop_pcd.normals = self.normals[pick_idx, :]

            return crop_pcd
        # get empty crop
        else:
            warnings.warn("Cropped 0 point in given polygon. Please check whether the coords is correct.")
            return None


def read_ply(ply_path):
    """Read the ply file

    Parameters
    ----------
    ply_path : str
        The path to las file

    Returns
    -------
    ndarray, ndarray, ndarray
        points, colors, normals of given point cloud data

    Example
    -------
    .. code-block:: python
    
        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> test_data.pcd.lotus_ply_asc
        WindowsPath('C:/Users/<User>/AppData/Local/easyidp.data/data_for_tests/pcd_test/hasu_tanashi_ascii.ply')
        >>> points, colors, normals = idp.pointcloud.read_ply(test_data.pcd.lotus_ply_asc)

        >>> points
        array([[-18.908312, -15.777558,  -0.77878 ],
               [-18.90828 , -15.777274,  -0.78026 ],
               [-18.907196, -15.774829,  -0.801748],
               ...,
               [-15.78929 , -17.96126 ,  -0.846867],
               [-15.788581, -17.939104,  -0.839632],
               [-15.786219, -17.936579,  -0.832714]], dtype=float32)
        >>> colors
        array([[123, 103,  79],
               [124, 104,  81],
               [123, 103,  80],
               ...,
               [116,  98,  80],
               [113,  95,  76],
               [115,  97,  78]], dtype=uint8)

    See also
    --------
    :func:`easyidp.PointCloud.read_point_cloud <easyidp.pointcloud.PointCloud.read_point_cloud>`
    """

    cloud_data = PlyData.read(ply_path).elements[0].data
    ply_names = cloud_data.dtype.names

    points = np.vstack((cloud_data['x'], cloud_data['y'], cloud_data['z'])).T

    if 'red' in ply_names:
        # range in 0-255
        colors = np.vstack((cloud_data['red'], cloud_data['green'], cloud_data['blue'])).T
    elif 'diffuse_red' in ply_names:
        colors = np.vstack((cloud_data['diffuse_red'], cloud_data['diffuse_green'], cloud_data['diffuse_blue'])).T
    else:
        print(f"Can not find color info in {ply_names}")
        colors = None

    colors.dtype = np.uint8

    # read normals
    if 'nx' in ply_names:
        normals = np.vstack((cloud_data['nx'], cloud_data['ny'], cloud_data['nz'])).T
    else:
        normals = None

    return points, colors, normals


def read_las(las_path):
    """Read the las file, the function wrapper for :func:`read_laz`

    Parameters
    ----------
    las_path : str
        The path to las file

    Returns
    -------
    ndarray, ndarray, ndarray
        points, colors, normals of given point cloud data

    Example
    -------
    .. code-block:: python
    
        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> test_data.pcd.lotus_ply_asc
        WindowsPath('C:/Users/<User>/AppData/Local/easyidp.data/data_for_tests/pcd_test/hasu_tanashi.las')
        >>> points, colors, normals = idp.pointcloud.read_las(test_data.pcd.lotus_las)

        >>> points
        array([[-18.908312, -15.777558,  -0.77878 ],
               [-18.90828 , -15.777274,  -0.78026 ],
               [-18.907196, -15.774829,  -0.801748],
               ...,
               [-15.78929 , -17.96126 ,  -0.846867],
               [-15.788581, -17.939104,  -0.839632],
               [-15.786219, -17.936579,  -0.832714]], dtype=float32)
        >>> colors
        array([[123, 103,  79],
               [124, 104,  81],
               [123, 103,  80],
               ...,
               [116,  98,  80],
               [113,  95,  76],
               [115,  97,  78]], dtype=uint8)

    See also
    --------
    :func:`easyidp.PointCloud.read_point_cloud <easyidp.pointcloud.PointCloud.read_point_cloud>`

    """
    return read_laz(las_path)


def read_laz(laz_path):
    """Read the laz file

    Parameters
    ----------
    laz_path : str
        The path to las file

    Returns
    -------
    ndarray, ndarray, ndarray
        points, colors, normals of given point cloud data

    Example
    -------
    .. code-block:: python
    
        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> test_data.pcd.lotus_ply_asc
        WindowsPath('C:/Users/<User>/AppData/Local/easyidp.data/data_for_tests/pcd_test/hasu_tanashi.laz')
        >>> points, colors, normals = idp.pointcloud.read_laz(test_data.pcd.lotus_laz)

        >>> points
        array([[-18.908312, -15.777558,  -0.77878 ],
               [-18.90828 , -15.777274,  -0.78026 ],
               [-18.907196, -15.774829,  -0.801748],
               ...,
               [-15.78929 , -17.96126 ,  -0.846867],
               [-15.788581, -17.939104,  -0.839632],
               [-15.786219, -17.936579,  -0.832714]], dtype=float32)
        >>> colors
        array([[123, 103,  79],
               [124, 104,  81],
               [123, 103,  80],
               ...,
               [116,  98,  80],
               [113,  95,  76],
               [115,  97,  78]], dtype=uint8)

    See also
    --------
    :func:`easyidp.PointCloud.read_point_cloud <easyidp.pointcloud.PointCloud.read_point_cloud>`
    
    """
    las = laspy.read(laz_path)

    points = np.vstack([las.x, las.y, las.z]).T

    # ranges 0-65536
    colors = np.vstack([las.points['red'], las.points['green'], las.points['blue']]).T / 256
    colors = colors.astype(np.uint8)

    # read normals
    """
    in cloudcompare, it locates "extended fields" -> normal x
    >>> las.point_format
    <PointFormat(2, 3 bytes of extra dims)>
    >>> las.point.array
    array([( 930206, 650950, 79707, 5654, 73, 0, 0, 0, 1, 7196,  5397, 4369, -4,  46, 118),
           ...,
           (1167278, 741188, 79668, 5397, 73, 0, 0, 0, 1, 6425,  5140, 4626, 41,  34, 114)],
          dtype=[('X', '<i4'), ('Y', '<i4'), ('Z', '<i4'), ('intensity', '<u2'), ('bit_fields', 'u1'), ('raw_classification', 'u1'), ('scan_angle_rank', 'i1'), ('user_data', 'u1'), ('point_source_id', '<u2'), ('red', '<u2'), ('green', '<u2'), ('blue', '<u2'), ('normal x', 'i1'), ('normal y', 'i1'), ('normal z', 'i1')])
    the normal type is int8 ('normal x', 'i1'), ('normal y', 'i1'), ('normal z', 'i1')
    but las.point['normal x'] -> get float value
    """
    if "normal x" in las.points.array.dtype.names:
        normals = np.vstack([las.points['normal x'], las.points['normal y'], las.points['normal z']]).T
    else:
        normals = None

    return points, colors, normals

def write_ply(ply_path, points, colors, normals=None, binary=True):
    """Save point cloud to ply format

    Parameters
    ----------
    ply_path : str
        the output point cloud file.
    points : ndarray
        the nx3 numpy ndarray of point XYZ info
    colors : ndarray
        the nx3 numpy ndarray of point RGB info, dtype=np.uint8
    normals :ndarray, optional
        the nx3 numpy ndarray of point normal info, by default None
    binary : bool, optional
        whether save the binary file.
        True: save BINARY ply file (by default)
        False: save ASCII ply file.

    Example
    -------
    Prepare data:

    .. code-block:: python
    
        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> write_points = np.asarray([[-1.9083118, -1.7775583,  -0.77878  ],
        ...                            [-1.9082794, -1.7772741,  -0.7802601],
        ...                            [-1.907196 , -1.7748289,  -0.8017483],
        ...                            [-1.7892904, -1.9612598,  -0.8468666],
        ...                            [-1.7885809, -1.9391041,  -0.839632 ],
        ...                            [-1.7862186, -1.9365788,  -0.8327141]], dtype=np.float64)
        >>> write_colors = np.asarray([[  0,   0,   0],
        ...                            [  0,   0,   0],
        ...                            [  0,   0,   0],
        ...                            [192,  64, 128],
        ...                            [ 92,  88,  83],
        ...                            [ 64,  64,  64]], dtype=np.uint8)
        >>> write_normals = np.asarray([[-0.03287353,  0.36604664,  0.9300157 ],
        ...                             [ 0.08860216,  0.07439037,  0.9932853 ],
        ...                             [-0.01135951,  0.2693031 ,  0.9629885 ],
        ...                             [ 0.4548034 , -0.15576138,  0.876865  ],
        ...                             [ 0.4550802 , -0.29450312,  0.8403392 ],
        ...                             [ 0.32758632,  0.27255052,  0.9046565 ]], dtype=np.float64)

    Use this function:

    .. code-block:: python

        >>> idp.pointcloud.write_ply(r"path/to/point_cloud.ply",  write_points, write_colors, binary=True)

    Notes
    -----
    (For developers)

    The ``plyfile`` packages requires to convert the ndarray outputs to numpy structured arrays [1]_ , then save 
    the point cloud structure looks like this:

    .. code-block:: python

        >>> cloud_data.elements
        (
            PlyElement(
                'vertex', 
                (
                    PlyProperty('x', 'float'), 
                    PlyProperty('y', 'float'), 
                    PlyProperty('z', 'float'), 
                    PlyProperty('red', 'uchar'), 
                    PlyProperty('green', 'uchar'), 
                    PlyProperty('blue', 'uchar')
                ), 
                count=42454, 
                comments=[]),
            )
        )
    
    convert ndarray to strucutred array [2]_ and method to merge to structured arrays [3]_

    References
    ----------
    .. [1] https://github.com/dranjan/python-plyfile#creating-a-ply-file
    .. [2] https://stackoverflow.com/questions/3622850/converting-a-2d-numpy-array-to-a-structured-array
    .. [3] https://stackoverflow.com/questions/5355744/numpy-joining-structured-arrays

    See also
    --------
    :func:`easyidp.PointCloud.write_point_cloud <easyidp.pointcloud.PointCloud.write_point_cloud>`

    """

    # convert to strucutrre array
    struct_points = np.core.records.fromarrays(points.T, names="x, y, z")
    struct_colors = np.core.records.fromarrays(colors.T, dtype=np.dtype([('red', np.uint8), ('green', np.uint8), ('blue', np.uint8)]))

    # add normals
    if normals is not None:
        struct_normals = np.core.records.fromarrays(normals.T, names="nx, ny, nz")
        merged_list = [struct_points, struct_colors, struct_normals]
    else:
        merged_list = [struct_points, struct_colors]

    # merge 
    struct_merge = rfn.merge_arrays(merged_list, flatten=True, usemask=False)

    # convert to PlyFile data type
    el = PlyElement.describe(struct_merge, 'vertex', 
                             comments=[f'Created by EasyIDP v{idp.__version__}', 
                                       f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}'])  

    # save to file
    if binary:
        PlyData([el]).write(ply_path)
    else:
        PlyData([el], text=True).write(ply_path)

def write_laz(laz_path, points, colors, normals=None, offset=np.array([0., 0., 0.]), decimal=5):
    """Save point cloud to laz format

    Parameters
    ----------
    laz_path : str
        the output point cloud file.
    points : ndarray
        the nx3 numpy ndarray of point XYZ info
    colors : ndarray
        the nx3 numpy ndarray of point RGB info, dtype=np.uint8
    normals :ndarray, optional
        the nx3 numpy ndarray of point normal info, by default None
    offset : 3x1 ndarray, optional
        The offset value defined in the laz file header, by default np.array([0., 0., 0.])
    decimal : int, optional
        The decimal for the point value precision, by default 5

    Example
    -------
    Prepare data:

    .. code-block:: python
    
        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> write_points = np.asarray([[-1.9083118, -1.7775583,  -0.77878  ],
        ...                            [-1.9082794, -1.7772741,  -0.7802601],
        ...                            [-1.907196 , -1.7748289,  -0.8017483],
        ...                            [-1.7892904, -1.9612598,  -0.8468666],
        ...                            [-1.7885809, -1.9391041,  -0.839632 ],
        ...                            [-1.7862186, -1.9365788,  -0.8327141]], dtype=np.float64)
        >>> write_colors = np.asarray([[  0,   0,   0],
        ...                            [  0,   0,   0],
        ...                            [  0,   0,   0],
        ...                            [192,  64, 128],
        ...                            [ 92,  88,  83],
        ...                            [ 64,  64,  64]], dtype=np.uint8)
        >>> write_normals = np.asarray([[-0.03287353,  0.36604664,  0.9300157 ],
        ...                             [ 0.08860216,  0.07439037,  0.9932853 ],
        ...                             [-0.01135951,  0.2693031 ,  0.9629885 ],
        ...                             [ 0.4548034 , -0.15576138,  0.876865  ],
        ...                             [ 0.4550802 , -0.29450312,  0.8403392 ],
        ...                             [ 0.32758632,  0.27255052,  0.9046565 ]], dtype=np.float64)

    Use this function:

    .. code-block:: python

        >>> idp.pointcloud.write_laz(r"path/to/point_cloud.laz", write_points, write_colors, write_normals)

    Notes
    -----
    .. caution::

        The EasyIDP saved the las file with Las version=1.2

    See also
    --------
    :func:`easyidp.PointCloud.write_point_cloud <easyidp.pointcloud.PointCloud.write_point_cloud>`
    """
    # create header
    header = laspy.LasHeader(point_format=2, version="1.2") 
    if normals is not None:
        header.add_extra_dim(laspy.ExtraBytesParams(name="normal x", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="normal y", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="normal z", type=np.float64))
    header.offsets = offset
    header.scales = np.array([float(f"1e-{decimal}")]*3)
    header.generating_software = f'EasyIDP v{idp.__version__} on {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}'

    # create las file
    las = laspy.LasData(header)

    # add values
    # in previous laspy, here has the convert to int32 precision loss
    # in laspy 2.5.4, dtype changed to float 64, fixed this change
    las.points['x'] = points[:, 0]  
    las.points['y'] = points[:, 1]   
    las.points['z'] = points[:, 2]

    # colors.dtype -> uint8; then convert to uint16
    #   numpy 2.0 -> uint8 can not convert to uint16 by `colors[:,0] * 256`
    #                need do astype first
    colors = colors.astype(np.uint16)
    las.points['red'] = colors[:,0] * 256    
    las.points['green'] = colors[:,1] * 256  
    las.points['blue'] = colors[:,2] * 256


    if normals is not None:
        las.points['normal x'] = normals[:, 0]
        las.points['normal y'] = normals[:, 1]
        las.points['normal z'] = normals[:, 2]

    las.write(laz_path)

def write_las(las_path, points, colors, normals=None, offset=np.array([0., 0., 0.]), decimal=5):
    """Save point cloud to las format, the function wrapper for :func:`write_laz`

    Parameters
    ----------
    las_path : str
        the output point cloud file.
    points : ndarray
        the nx3 numpy ndarray of point XYZ info
    colors : ndarray
        the nx3 numpy ndarray of point RGB info, dtype=np.uint8
    normals :ndarray, optional
        the nx3 numpy ndarray of point normal info, by default None
    offset : 3x1 ndarray, optional
        The offset value defined in the laz file header, by default np.array([0., 0., 0.])
    decimal : int, optional
        The decimal for the point value precision, by default 5

    Example
    -------
    Prepare data:

    .. code-block:: python
    
        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> write_points = np.asarray([[-1.9083118, -1.7775583,  -0.77878  ],
        ...                            [-1.9082794, -1.7772741,  -0.7802601],
        ...                            [-1.907196 , -1.7748289,  -0.8017483],
        ...                            [-1.7892904, -1.9612598,  -0.8468666],
        ...                            [-1.7885809, -1.9391041,  -0.839632 ],
        ...                            [-1.7862186, -1.9365788,  -0.8327141]], dtype=np.float64)
        >>> write_colors = np.asarray([[  0,   0,   0],
        ...                            [  0,   0,   0],
        ...                            [  0,   0,   0],
        ...                            [192,  64, 128],
        ...                            [ 92,  88,  83],
        ...                            [ 64,  64,  64]], dtype=np.uint8)
        >>> write_normals = np.asarray([[-0.03287353,  0.36604664,  0.9300157 ],
        ...                             [ 0.08860216,  0.07439037,  0.9932853 ],
        ...                             [-0.01135951,  0.2693031 ,  0.9629885 ],
        ...                             [ 0.4548034 , -0.15576138,  0.876865  ],
        ...                             [ 0.4550802 , -0.29450312,  0.8403392 ],
        ...                             [ 0.32758632,  0.27255052,  0.9046565 ]], dtype=np.float64)

    Use this function:

    .. code-block:: python

        >>> idp.pointcloud.write_las(r"path/to/point_cloud.las", write_points, write_colors, write_normals)

    Notes
    -----
    .. caution::

        The EasyIDP saved the las file with Las version=1.2

    See also
    --------
    :func:`easyidp.PointCloud.write_point_cloud <easyidp.pointcloud.PointCloud.write_point_cloud>`
    
    """
    write_laz(las_path, points, colors, normals, offset, decimal)