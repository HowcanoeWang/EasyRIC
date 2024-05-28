import os
import pyproj
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from skimage.io import imread, imsave

import easyidp as idp

class ProjectPool(idp.Container):

    def __init__(self) -> None:
        super().__init__()
        self.id_item = {}
        self.item_label = {}

    def add_pix4d(self, paths):
        # proj.add_pix4d(["aaa.p4d", "bbb.p4d", ...]) 
        pass

    def add_metashape(self, paths):
        # proj.add_metashape(["aaa.psx", "bbb.psx"]) support using list to give time-series data
        pass


class Recons(object):
    """
    The base class for reconstruction project. Used for each individual Pix4D project and each chunk in Metashape project

    .. note::

        Coordinate systems used in the 3D reconstruction.

        - internal coordinate (local): 
        
            the coordinate used in current chunk, often the center of model as initial point

        - geocentric coordinate (world): 
        
            use the earth's core as initial point, also called world coordinate

        - geographic coordinate (crs):
        
            coordinate reference system (CRS) to locate geographical entities. Common used:
        
            - ``WGS84 (EPSG: 4326)``: xyz = longitude, latitude, altitude
            - ``WGS84/ UTM Zone xxx``: e.g. UTM Zone 54N -> Tokyo area.

    """
    def __init__(self):

        #: the 3D reconstruction project name, ``<class 'str'>``
        self.label = ""
        #: meta information in this project, ``<class 'dict'>``
        self.meta = {}
        #: whether this project is activated, (often for Metashape), ``<class 'bool'>``
        self.enabled = True

        #: the container for all sensors in this project (camera model), ``<class 'easyidp.Container'>``, a dict-like object consisted by :py:class:`Sensor <easyidp.reconstruct.Sensor>`
        self.sensors = idp.Container()
        #: the container for all photos used in this project (images), ``<class 'easyidp.Container'>``, a dict-like object consisted by :py:class:`Photo <easyidp.reconstruct.Photo>`
        self.photos = idp.Container()

        #: the world crs for geocentric coordiante, ``<class 'pyproj.crs.crs.CRS'>``
        self._world_crs = pyproj.CRS.from_dict({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'})

        self._dom = idp.GeoTiff()
        self._dsm = idp.GeoTiff()
        self._pcd = idp.PointCloud()
        self._crs = None

        self._photo_position_cache = None

    @property
    def crs(self):
        """the geographic coordinates (often the same as the export DOM and DSM),  ``<class 'pyproj.crs.crs.CRS'>``"""
        return self._crs

    @crs.setter
    def crs(self, new_crs):
        # clear photo position cache once changed the project crs
        if isinstance(new_crs, pyproj.CRS) or new_crs is None:
            self._photo_position_cache = None
            self._crs = new_crs
        else:
            raise TypeError(f"Only <'pyproj.CRS' object> is acceptable for property .crs, not given {type(new_crs)}")

    @property
    def dom(self):
        """The output digitial orthomosaic map (DOM), :class:`easyidp.GeoTiff <easyidp.geotiff.GeoTiff>`"""
        # default None
        if self._dom.has_data():
            return self._dom
        else:
            return None

    @dom.setter
    def dom(self, p):
        if isinstance(p, (Path, str)):
            if Path(p).exists():
                self._dom.read_geotiff(p)
            else:
                raise FileNotFoundError(f"Given DOM file [{p}] does not exists")
        elif isinstance(p, idp.GeoTiff):
            self._dom = p
        else:
            raise TypeError(f"Please either specify DOM file path (str) or idp.GeoTiff objects, not {type(p)}")

    @property
    def dsm(self):
        """The output digitial surface map (DSM), :class:`easyidp.GeoTiff <easyidp.geotiff.GeoTiff>`"""
        if self._dsm.has_data():
            return self._dsm
        else:
            return None

    @dsm.setter
    def dsm(self, p):
        if isinstance(p, (Path, str)):
            if Path(p).exists():
                self._dsm.read_geotiff(p)
            else:
                raise FileNotFoundError(f"Given DSM file [{p}] does not exists")
        elif isinstance(p, idp.GeoTiff):
            self._dsm = p
        else:
            raise TypeError(f"Please either specify DSM file path (str) or idp.GeoTiff objects, not {type(p)}")

    @property
    def pcd(self):
        """The output point cloud, :class:`easyidp.PointCloud <easyidp.pointcloud.PointCloud>`"""
        if self._pcd.has_points():
            return self._pcd
        else:
            return None

    @pcd.setter
    def pcd(self, p):
        if isinstance(p, (Path, str)):
            if Path(p).exists():
                self._pcd.read_point_cloud(p)
            else:
                raise FileNotFoundError(f"Given pointcloud file [{p}] does not exists")
        elif isinstance(p, idp.PointCloud):
            self._pcd = p
        else:
            raise TypeError(f"Please either specify pointcloud file path (str) or idp.PointCloud objects, not {type(p)}")


class Sensor:
    """The base class of camera model"""

    def __init__(self):
        #: the sensor id in this 3D reconstruction project, often only has one. ``<class 'int'>``
        self.id = 0
        #: the sensor label/name, ``<class 'str'>``
        self.label = ""
        #: Sensor type in [frame, fisheye, spherical, rpc] (often for metashape project), ``<class 'str'>``
        self.type = "frame"
        #: The sensor width pixel number, ``<class 'int'>``
        self.width = 0  # in int
        #: The sensor height pixel number, ``<class 'int'>``
        self.height = 0  # in int

        #: sensor actual width, unit is mm, ``<class 'float'>``
        self.w_mm = 0.0
        #: sensor actual height, unit is mm, ``<class 'float'>``
        self.h_mm = 0.0
        
        #: the scale of one pixel width, unit in mm, ``<class 'float'>``
        self.pixel_width = 0.0
        #: the scale of one pixel height, unit in mm, ``<class 'float'>``
        self.pixel_height = 0.0
        #: the scale of one pixel, for pix4d, [pixel_height, pixel_width]
        self.pixel_size = []

        #: focal length, unit in mm, ``<class 'float'>``
        self.focal_length = 0.0  # in mm

        #: sensor calibration information, :py:class:`easyidp.Calibration <easyidp.reconstruct.Calibration>`
        self.calibration = Calibration(self)

    def in_img_boundary(self, polygon_hv, ignore=None, log=False):
        """Judge whether given polygon is in the image area, and move points to boundary if specify ignore.

        Parameters
        ----------
        polygon_hv : numpy nx2 array
            [horizontal, vertical] points in pixel coordinate
        ignore : str | None, optional
            Whether tolerate small parts outside image

            - ``None``: strickly in image area;
            - ``x``: only y (vertical) in image area, x can outside image;
            - ``y``: only x (horizontal) in image area, y can outside image.

            .. todo::

                This API will be enhanced and changed in the future.

                ``ignore`` (str) -> ``ignore_overflow`` (bool):

                - ``True``: strickly in image area, default;
                - ``False``: cut the polygon inside the image range;
                
                .. image:: ../../_static/images/python_api/back2raw_ignore_todo.png
                    :alt: back2raw_ignore_todo.png'
                    :scale: 60
        log : bool, optional
            whether print log for debugging, by default False

        Returns
        -------
        None | polygon_hv
        """
        w, h = self.width, self.height
        coord_min = polygon_hv.min(axis=0)
        x_min, y_min = coord_min[0], coord_min[1]
        coord_max= polygon_hv.max(axis=0)
        x_max, y_max = coord_max[0], coord_max[1]

        if ignore is None:
            if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
                if log: print(f'X  w[{x_min} ~ {x_max}], h[{y_min} ~ {y_max}]')
                return None
            else:
                if log: print(f'O  w[{x_min} ~ {x_max}], h[{y_min} ~ {y_max}]')
                return polygon_hv
        elif ignore=='x':
            warnings.warn(
                "This API `ignore` (str) will be enhanced and "
                "changed to `ignore_overflow` (bool) in the future.", 
                FutureWarning
            )
            if y_min < 0 or y_max > h:
                if log: print(f'X  w[{x_min}-{x_max}], h[{y_min}-{y_max}]')
                return None
            else:
                # replace outside points to image boundary
                polygon_hv[polygon_hv[:, 0] < 0, 0] = 0
                polygon_hv[polygon_hv[:, 0] > w, 0] = w
                if log: print(f'O  w[{x_min}-{x_max}], h[{y_min}-{y_max}]')
                return polygon_hv
        elif ignore=='y':
            warnings.warn(
                "This API `ignore` (str) will be enhanced and "
                "changed to `ignore_overflow` (bool) in the future.", 
                FutureWarning
            )
            if x_min < 0 or x_max > w:
                if log: print(f'X  w[{x_min}-{x_max}], h[{y_min}-{y_max}]')
                return None
            else:
                # replace outside point to image boundary
                polygon_hv[polygon_hv[:, 1] < 0, 1] = 0
                polygon_hv[polygon_hv[:, 1] > h, 1] = h
                if log: print(f'O  w[{x_min}-{x_max}], h[{y_min}-{y_max}]')
                return polygon_hv
        elif ignore=='as_point':
            # Convert polygon to points, then just keep the points inside images.
            # This mode is designed for broccoli position points backward projection or UAVBroccoli project
            points_left = polygon_hv[(polygon_hv[:,0] > 0) & (polygon_hv[:,0] < w) & 
                                     (polygon_hv[:,1] > 0) & (polygon_hv[:,1] < h), :]
            return points_left
        else:
            raise ValueError(f"`ignore` should be None, 'x', or 'y', not {ignore}")


class Photo:
    """The base class to store image information used in 3D reconstruction project"""

    # Modifed from the old API for pix4d
    # 
    # class Image:
    #     def __init__(self, name, path, w, h, pmat, cam_matrix, rad_distort, tan_distort, cam_pos, cam_rot):
    #         # external parameters
    #         self.name = name   -> label
    #         self.path = path
    #         self.w = w    -> sensor property
    #         self.h = h    -> sensor property
    #         self.pmat = pmat   # transform matrix? 3x4 -> 4x4
    #         self.cam_matrix = cam_matrix
    #         #self.rad_distort = rad_distort   # seems not used
    #         #self.tan_distort = tan_distort
    #         self.cam_pos = cam_pos   -> location
    #         self.cam_rot = cam_rot   -> rotation

    def __init__(self, sensor=None):
        #: The id of current image in reconstruction project, ``<class 'int'>``
        self.id = 0

        #: The image path in local computer, ``<class 'str'>``
        self.path = ""
        self._path = ""   # this is the relative path, often stored by metashape project

        #: the image name, ``<class 'str'>``
        self.label = ""

        #: the id of the camera model (sonsor), ``<class 'int'>``
        self.sensor_id = 0
        #: the object of the camera model (sensor), :class:`easyidp.Sensor <easyidp.reconstruct.Sensor>`
        self.sensor = sensor

        #: whether this image is used in the 3D reconstruction,  ``<class 'bool'>``
        self.enabled = True

        # reconstruction info in local coord
        #: the 3x3 camera matrix, the ``K`` in ``K[R t]``, ``<class 'numpy.ndarray'>``
        self.cam_matrix = None # np.zeros(3,3) -> K
        #: the 3x1 vector of camera location, the ``t`` in ``K[R t]``, ``<class 'numpy.ndarray'>``
        self.location = None  # np.zeros(3,) -> t
        #: the 3x3 rotation matrix, the ``R`` in ``K[R t]``, ``<class 'numpy.ndarray'>``
        self.rotation = None  # np.zeros(3,3) -> R

        #: the transform matrix, different between pix4d and metashape project, please check below for more details ``<class 'numpy.ndarray'>``
        #:
        #: - In ``metashape``: it is the 4x4 matrix describing photo location in the chunk coordinate system -> ``K[R t]``
        #: - in ``pix4d``: it is the 3x4 pmatrix. Please check `Pix4D PMatrix documentation <https://support.pix4d.com/hc/en-us/articles/202977149-What-does-the-Output-Params-Folder-contain#label12>` for more details
        #:
        self.transform = None  # 
        #: the 3x1 translation vector, often provided by metashape.
        self.translation = None  # np.zeros(3,)

        # output infomation
        #: The 3x1 vector of geo coodinate of image in real world, ``<class 'numpy.ndarray'>``
        self.position = None # -> in outputs geo_coordiantes

        # meta info, not necessary in current version
        #self.time = ""
        #self.gps = {"altitude": 0.0, "latitude": 0.0, "longitude": 0.0}
        #self.xyz = {"X": 0, "Y": 0, "Z": 0}
        #self.orientation = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        # parent info, for multispectral cameras
        self.master_id = None

    def _img_exists(func):
        """the decorator to check if image exists"""
        def wrapper(self, *args, **kwargs):
            if self.path != "" or not os.path.exists(self.path):
                raise FileNotFoundError("Could not operate if not specify correct image file path")
            return func(self, *args, **kwargs)

        return wrapper

    # @_img_exists
    # def get_imarray(self, roi=None):
    #     """Read the original image, and return a image array, not implemented yet"""

    #     raise NotImplementedError("This function has not been fully supported yet")


class Calibration:
    """The base class for camera lens distortion calibration"""

    def __init__(self, sensor=None):
        #: the calibration model from which reconstruction software, in ["pix4d", "metashape"], ``<class 'str'>``
        self.software = "metashape"
        #: the calibration type, same as the sensor.type, in [frame, fisheye, spherical, rpc], by default 'frame'
        self.type = "frame"
        #: the object of the camera model (sensor), :py:class:`Sensor <easyidp.reconstruct.Sensor>`
        self.sensor = sensor

        #: focal length, unit is pixel, for pix4d project, convert mm to pixel. ``<class 'float'>``
        self.f = 0.0 
        
        #: principle point offset, unit is pixel.
        #: 
        #: .. note::
        #:    In the older version of metashape, Cx and Cy were given in pixels from the top-left corner of the image.
        #:    But in the latest release version they are measured as offset from the image center.
        #:    Reference: `https://www.agisoft.com/forum/index.php?topic=5827.0``
        #:
        self.cx = 0.0
        #: principle point offset, unit is pixel.
        self.cy = 0.0

        #: affinity and non-orthogonality (skew) coefficients (in pixels) [metashape use only] 
        self.b1 = 0.0
        #: affinity and non-orthogonality (skew) coefficients (in pixels) [metashape use only]
        self.b2 = 0.0

        #: len distortion coefficient, different between pix4d and metashape, please check below for more details
        #:
        #: - ``pix4d``: Symmetrical Lens Distortion Coeffs
        #: - ``metashape``: radial distortion coefficients (dimensionless)
        self.k1 = 0.0
        #: len distortion coefficient
        self.k2 = 0.0
        #: len distortion coefficient
        self.k3 = 0.0
        #: len distortion coefficient
        self.k4 = 0.0

        #: Tangential Lens Distortion Coefficients, for Pix4D.
        #:
        #: .. note::
        #:
        #:    - ``pix4d``: Tangential Lens Distortion Coeffs, use T
        #:    - ``metashape``: tangential distortion coefficient, use P
        #:
        self.t1 = 0.0  # metashape -> p1
        #: Tangential Lens Distortion Coeffs
        self.t2 = 0.0  # metashape -> p2
        #: Tangential Lens Distortion Coeffs
        self.t3 = 0.0  # metashape -> p3
        #: Tangential Lens Distortion Coeffs
        self.t4 = 0.0  # metashape -> p4

        #: Tangential Lens Distortion Coefficients, for Metashape.
        #:
        #: .. note::
        #:
        #:    - ``pix4d``: Tangential Lens Distortion Coeffs, use T
        #:    - ``metashape``: tangential distortion coefficient, use P
        #:
        self.p1 = self.t1  # metashape -> p1
        #: Tangential Lens Distortion Coeffs
        self.p2 = self.t2  # metashape -> p2
        #: Tangential Lens Distortion Coeffs
        self.p3 = self.t3  # metashape -> p3
        #: Tangential Lens Distortion Coeffs
        self.p4 = self.t4  # metashape -> p4

    def calibrate(self, u, v):
        """Convert undistorted images -> original image pixel coordinate

        Parameters
        ----------
        u : ndarray
            the x pixel coordinate after R transform
        v : ndarray
            the y pixel coordinate after R transform

        Returns
        -------
        xb, yb: 
            the pixel coordinate on the original image


        Note
        ----

        The calculation formular can be references by :

        - Pix4D: #2.1.2 section in [1]_ .
        - Metashape: Appendix C. Camera models in [2]_ .


        References
        ----------
        .. [1] https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
        .. [2] https://www.agisoft.com/pdf/metashape-pro_1_7_en.pdf

        """
        if self.software == "pix4d":
            if self.type == "frame":
                return self._calibrate_pix4d_frame(u, v)
            else:
                raise NotImplementedError(
                    f"Can not calibrate camera type [{self.type}], "
                    "only support [frame] currently"
                )
        elif self.software == "metashape":
            if self.type == "frame":
                return self._calibrate_metashape_frame(u, v)
            else:
                raise NotImplementedError(
                    f"Can not calibrate camera type [{self.type}], "
                    "only support [frame] currently"
                )
        else:
            raise TypeError(
                f"Could only handle [pix4d | metashape] projects, "
                "not {self.type}")

    def _calibrate_pix4d_frame(self, u, v):
        """Convert undistorted images -> original image pixel coordinate

        Parameters
        ----------
        u : ndarray
            the x pixel coordinate after R transform
        v : ndarray
            the y pixel coordinate after R transform

        Returns
        -------
        xb, yb: 
            the pixel coordinate on the original image

        Notes
        -----
        Formula please refer: #2.1.2 in
        https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
        """
        f = self.f
        cx = self.cx
        cy = self.cy

        xh = (u - cx) / f
        yh = (v - cy) / f

        r2 = xh ** 2 + yh ** 2
        r4 = r2 ** 2
        r6 = r2 ** 3
        a1 = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        xhd = a1 * xh + 2 * self.t1 * xh * yh + self.t2 * (r2 + 2 * xh ** 2)
        yhd = a1 * yh + 2 * self.t2 * xh * yh + self.t1 * (r2 + 2 * yh ** 2)

        xh = f * xhd + cx
        yh = f * yhd + cy

        return xh, yh

    def _calibrate_metashape_frame(self, xh, yh):
        """Convert undistorted images -> original image pixel coordinate

        Parameters
        ----------
        xh : ndarray
            the x pixel coordinate after R transform
        yh : ndarray
            the y pixel coordinate after R transform

        Returns
        -------
        xb, yb: 
            the pixel coordinate on the original image

        Notes
        -----
        Formula please refer: Appendix C. Camera models
        https://www.agisoft.com/pdf/metashape-pro_1_7_en.pdf
        """
        r2 = xh ** 2 + yh ** 2
        r4 = r2 ** 2
        r6 = r2 ** 3
        r8 = r2 ** 4

        f = self.f
        cx = self.cx
        cy = self.cy

        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4

        p1 = self.t1
        p2 = self.t2
        b1 = self.b1
        b2 = self.b2

        w = self.sensor.width
        h = self.sensor.height

        eq_part1 = (1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8)

        x_prime = xh * eq_part1 + (p1 * (r2 + 2 * xh ** 2) + 2 * p2 * xh * yh)
        y_prime = yh * eq_part1 + (p2 * (r2 + 2 * yh ** 2) + 2 * p1 * xh * yh)

        xb = w * 0.5 + cx + x_prime * f + x_prime * b1 + y_prime * b2
        yb = h * 0.5 + cy + y_prime * f

        return xb, yb


class ChunkTransform:
    """Similar API wrapper for Metashape Python API ``class Metashape.ChunkTransform``
    """

    def __init__(self):

        #: Transformation matrix
        self.matrix = None
        #: Rotation compone
        self.rotation = None
        #: Translation compone
        self.translation = None
        #: Scale compone
        self.scale = None
        #: Inverse matrix
        #:
        #: .. note::
        #: 
        #:     Inspired from Kunihiro Kodama's Metashape API usage <kkodama@kazusa.or.jp>
        #: 
        #:     .. code-block:: python
        #: 
        #:         >>> import Metashape
        #:         >>> chunk = Metashape.app.document.chunk()
        #:         >>> transm = chunk.transform.matrix
        #:         >>> invm = Metashape.Matrix.inv(chunk.transform.matrix)
        #: 
        #:     invm.mulp(local_vec) --> transform chunk local coord to world coord (if you handle vec in local coord)
        #: 
        #:     How to calculate from xml data: `Agisoft Forum: Topic: Camera coordinates to world <https://www.agisoft.com/forum/index.php?topic=6176.0>`_
        #: 
        self.matrix_inv = None


def _sort_img_by_distance_one_roi(recons, img_dict, plot_geo, cam_pos, distance_thresh=None, num=None):
    """Sort the back2raw img_dict results by distance from photo to roi

    Parameters
    ----------
    recons: idp.Metashape or idp.Pix4D
        The reconsturction project class
    img_dict : dict
        One ROI output dict of roi.back2raw()
        e.g. img_dict = roi.back2raw(ms) -> img_dict["N1W1"]
    plot_geo : nx3 ndarray
        The plot boundary polygon vertex coordinates
    num : None or int
        Keep the closest {x} images
    distance_thresh : None or float
        If given, filter the images smaller than this distance first

    Returns
    -------
    dict
        the same structure as output of roi.back2raw()
    """
    dist_geo = []
    dist_name = []

    img_dict_sort = {}

    for img_name in img_dict.keys():
        xmin_geo, ymin_geo = plot_geo[:,0:2].min(axis=0)
        xmax_geo, ymax_geo = plot_geo[:,0:2].max(axis=0)

        xctr_geo = (xmax_geo + xmin_geo) / 2
        yctr_geo = (ymax_geo + ymin_geo) / 2

        ximg_geo, yimg_geo, _ = cam_pos[img_name]

        image_plot_dist = np.sqrt((ximg_geo-xctr_geo) ** 2 + (yimg_geo - yctr_geo) ** 2)

        if distance_thresh is not None and image_plot_dist > distance_thresh:
            # skip those image-plot geo distance greater than threshold
            continue
        else:
            # if not given dist_thresh, record all
            dist_geo.append(image_plot_dist)
            dist_name.append(img_name)

    if num is None:
        # not specify num, use all
        num = len(dist_name)
    else:
        num = min(len(dist_name), num)

    dist_geo_idx = np.asarray(dist_geo).argsort()[:num]
    img_dict_sort = {dist_name[idx]:img_dict[dist_name[idx]] for idx in dist_geo_idx}

    return img_dict_sort


def sort_img_by_distance(recons, img_dict_all, roi, distance_thresh=None, num=None, save_folder=None):
    """Advanced wrapper of sorting back2raw img_dict results by distance from photo to roi

    Parameters
    ----------
    recons: idp.Metashape or idp.Pix4D
        The reconsturction project class
    img_dict_all : dict
        All output dict of roi.back2raw(...)
        e.g. img_dict = roi.back2raw(...) -> img_dict
    roi: idp.ROI
        Your roi variable
    num : None or int
        Keep the closest {x} images
    distance_thresh : None or float
        Keep the images closer than this distance to ROI.
    save_folder : str, optional
        the folder to save json files and parts of ROI on raw images, by default None

    Returns
    -------
    dict
        the same structure as output of roi.back2raw()
    """
    cam_pos = recons.get_photo_position(to_crs=roi.crs)

    img_dict_sort_all = {}
    pbar = tqdm(roi.keys(), desc=f"Filter by distance to ROI")
    for roi_name in pbar:
        sort_dict = _sort_img_by_distance_one_roi(
            recons, img_dict_all[roi_name], roi[roi_name], 
            cam_pos, distance_thresh, num
        )
        img_dict_sort_all[roi_name] = sort_dict

    if save_folder is not None:
        save_back2raw_json_and_png(recons, img_dict_sort_all, save_folder)

    return img_dict_sort_all


def save_back2raw_json_and_png(recons, results_dict, save_folder):
    """Save the backward reversed results 

    Parameters
    ----------
    results_dict : dict
        the outputs of :func:`back2raw() <easyidp.roi.back2raw()>` function results
    save_folder : str
        the folder to save output files

    Example
    -------

    Data prepare

    .. code-block:: python

        >>> import easyidp as idp
        >>> lotus = idp.data.Lotus()
        >>> p4d = idp.Pix4D(lotus.pix4d.project, lotus.photo, lotus.pix4d.param)

        >>> roi = idp.ROI(lotus.shp, name_field=0)
        >>> roi = roi[0:3]
        >>> roi.get_z_from_dsm(lotus.pix4d.dsm)

        >>> out_p4d = roi.back2raw(p4d)

    Use this function:

    .. code-block:: python

        >>> idp.reconstruct.save_back2raw_json_and_png(p4d, out_p4d, "out_path")

    """
    # prepare the root folder
    if isinstance(save_folder, (str, Path)):
        save_folder = str(save_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    else:
        raise TypeError(f"Only the string path is acceptable, not [{save_folder} {type(save_folder)}]")
    
    # reverse the current order results_dict[roi][image_name] to results[image_name][roi]
    # this will save the IO cost when loading images.
    print("Optimising data structures of produced results, this may take some time...")
    rev_dict = {}
    for roi_name, roi_dict in results_dict.items():
        # create the save path of each roi
        roi_folder = os.path.join(save_folder, roi_name)
        if not os.path.exists(roi_folder):
            os.mkdir(roi_folder)

        for img_name, img_pos in roi_dict.items():
            if img_name not in rev_dict.keys():
                rev_dict[img_name] = {}

            rev_dict[img_name][roi_name] = img_pos

    # save the full results as json directly
    idp.jsonfile.save_json(results_dict, os.path.join(save_folder, "roi_image_order.json"))
    idp.jsonfile.save_json(rev_dict, os.path.join(save_folder, "image_roi_order.json"))

    # then doing the for loop for each image.
    for img_name, img_rois in tqdm(rev_dict.items(), desc=f"Processing image [{img_name}]"):

        img_array = imread(recons.photos[img_name].path)

        for roi_name, img_pos in tqdm(img_rois.items(), leave=False):

            cropped_png, off = idp.cvtools.imarray_crop(img_array, img_pos)

            png_save_path = os.path.join(save_folder, roi_name, f"{roi_name}_{img_name}_at_top_{off[0]}_left_{off[1]}.png")

            imsave(png_save_path, cropped_png)