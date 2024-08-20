import os
import numpy as np
import warnings
import pyproj
from pathlib import Path
from tqdm import tqdm
from copy import copy as ccopy

import easyidp as idp

class Pix4D(idp.reconstruct.Recons):
    """A Pix4D class, contains information of 3D reconstruction."""

    def __init__(self, project_path=None, raw_img_folder=None, param_folder=None):
        """The method to initialize the Pix4D class

        Parameters
        ----------
        project_path : str, optional
            The pix4d project file to open, like "xxxx.p4d", by default None, means create an empty class
        raw_img_folder : str, optional
            the original UAV image folder, by default None
        param_folder : str, optional
            the folder of pix4d project parameters, just in case user changed the default folder structure 
            (``...\\project_name\\1_initial\\params\\``), by default None

        Example
        -------

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

        Then open the demo pix4d project:

        .. code-block:: python

            >>> p4d = idp.Pix4D(test_data.pix4d.maize_folder)


        Or manual specify parameters if the project folder structure has been changed. 

        .. code-block:: python

            >>> p4d = idp.Pix4D(
            ...     project_path   = test_data.pix4d.lotus_folder, 
            ...     raw_img_folder = test_data.pix4d.lotus_photos, 
            ...     param_folder   = test_data.pix4d.lotus_param
            ... )

        Or you can create an empty project, and the open a given path:

        .. code-block:: python

            >>> p4d = idp.Pix4D()
            >>> p4d.open_project(
            ...     project_path   = test_data.pix4d.lotus_folder, 
            ...     raw_img_folder = test_data.pix4d.lotus_photos, 
            ...     param_folder   = test_data.pix4d.lotus_param
            ... )

        .. caution::

            In previous case, the manager reorganized the project structure and outputs of ``test_data.pix4d.lotus_folder``
            
            (e.g., moved the ``\\project_name\\1_initial\\params\\`` to ``\\project_name\\params\\``, 
            as well as other outputs, the following folder is no more a standard pix4d project)

            .. code-block:: bash

                $ls '/Users/<user>/Library/Application Support/easyidp.data/data_for_tests/pix4d/lotus_tanashi_full'

                params/
                photos/
                hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif
                hasu_tanashi_20170525_Ins1RGB_30m_group1_densified_point_cloud.ply
                hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif
                plot_dom.tif
                plot_dsm.tif
                plot_pcd.ply

            The default loading doesn't work, because it is not a standard pix4d project:

            .. code-block:: python

                >>> p4d = idp.Pix4D(test_data.pix4d.lotus_folder)

                Traceback (most recent call last):
                    File "<stdin>", line 1, in <module>
                    File "/Users/hwang/OneDrive/Program/GitHub/EasyIDP/easyidp/pix4d.py", line 48, in __init__
                    File "/Users/hwang/OneDrive/Program/GitHub/EasyIDP/easyidp/pix4d.py", line 56, in open_project
                        hasu_tanashi_20170525_Ins1RGB_30m_group1_densified_point_cloud.ply
                    File "/Users/hwang/OneDrive/Program/GitHub/EasyIDP/easyidp/pix4d.py", line 829, in parse_p4d_project
                        sub_folder = os.listdir(project_path)
                FileNotFoundError: Can not find pix4d parameter in given project folder
                

            In this case, must manual specfiy the ``param_folder``
        
        """
        super().__init__()

        #: the 3D reconstruction project software, in ['pix4d', 'metashape'], ``<class 'str'>``
        self.software = "pix4d"
        #: pix4d point cloud offset
        self.offset_np = np.zeros((3,1))


        ########################################
        # mute attributes warning for auto doc #
        ########################################
        #: project / chunk name
        self.label = self.label
        #: project meta information
        self.meta = self.meta
        #: whether this project is activated, (often for Metashape), ``<class 'bool'>``
        self.enabled = self.enabled
        #: the container for all sensors in this project (camera model), ``<class 'easyidp.Container'>``
        self.sensors = self.sensors
        #: the container for all photos used in this project (images), ``<class 'easyidp.Container'>``
        self.photos = self.photos

        if project_path is not None:
            self.open_project(project_path, raw_img_folder, param_folder)


    def open_project(self, project_path, raw_img_folder=None, param_folder=None):
        """Open a new 3D reconstructin project to overwritting current project.

        Parameters
        ----------
        project_path : str
            The pix4d project file to open, like "xxxx.p4d", or "xxxx" without suffix
        raw_img_folder : str, optional
            the original UAV image folder, by default None
        param_folder : str, optional
            the folder of pix4d project parameters, just in case user changed the default folder structure 
            (``...\\project_name\\1_initial\\params\\``), by default None

        Example
        -------

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

        Then using this function to open a new project:

        .. code-block:: python

            >>> p4d = idp.Pix4D()
            >>> p4d.open_project(
            ...     project_path   = test_data.pix4d.lotus_folder, 
            ...     raw_img_folder = test_data.pix4d.lotus_photos, 
            ...     param_folder   = test_data.pix4d.lotus_param
            ... )

        """
        project_path = str(project_path)
        # check if project_path = xxxx.p4d
        if ".p4d" == project_path[-4:]:
            project_path = project_path[:-4]

        p4d_dict = parse_p4d_project(project_path, param_folder)

        #: project / chunk name
        self.label = p4d_dict["project_name"]

        #: project meta information
        self.meta["p4d_offset"] = read_xyz(p4d_dict["param"]["xyz"])
        #: pix4d point cloud offset
        self.offset_np = self.meta["p4d_offset"]

        #####################
        # info for Sensor() #
        #####################
        ccp = read_ccp(p4d_dict["param"]["ccp"])
        cicp = read_cicp(p4d_dict["param"]["cicp"])
        ssk = read_cam_ssk(p4d_dict["param"]["ssk"])
        '''
        CCP:
        {'w': 4608, 
         'h': 3456, 
         'Image1.JPG': 
            {'cam_matrix': array([[...]]),   # (3x3) K
             'rad_distort': array([ 0.03833474, ...02049799]),
             'tan_distort': array([0.00240852, 0...00292562]), 
             'cam_pos': array([ 21.54872207,...8570281 ]),  (3x1) t
             'cam_rot': array([[ 0.78389904,...99236  ]])}, (3x3) R
             
         'Image2.JPG':
            ...
        }


        CICP:
        {'w_mm','h_mm', 'F', 'Px', 'Py', 'K1', 'K2', 'K3', 'T1', 'T2'}
        #Focal Length mm assuming a sensor width of 17.49998592000000030566x13.12498944000000200560mm
        F 15.01175404934517487732
        #Principal Point mm
        Px 8.48210511970419922534
        Py 6.33434629978042273990
        #Symmetrical Lens Distortion Coeffs
        K1 0.03833474118270804865
        K2 -0.01750917966495743258
        K3 0.02049798716391852335
        #Tangential Lens Distortion Coeffs
        T1 0.00240851666319534747
        T2 0.00292562392135245920
        '''

        sensor = idp.reconstruct.Sensor()
        # seems pix4d only support one camera kind
        sensor.id = 0
        sensor.label = ssk["label"]
        sensor.type = ssk["type"]

        sensor.h_mm = cicp["h_mm"]
        sensor.w_mm = cicp["w_mm"]
        sensor.focal_length = cicp["F"]

        sensor.height = ccp["h"]
        sensor.width = ccp["w"]

        # ensure the order is correct
        # ssk -> image_size_in_pixels: 3456 4608
        # it should be the orientation == 1
        if ssk["image_size_in_pixels"] == [ccp["h"], ccp["w"]]:
            sensor.pixel_size = ssk["pixel_size"]
            # the scale of each pixel, unit is mm
            sensor.pixel_height, sensor.pixel_width = ssk["pixel_size"]

            # calibration params
            sensor.calibration.cy, sensor.calibration.cx = ssk["photo_center_in_pixels"]
        # the order is reversed, probably orientatio == 0
        elif ssk["image_size_in_pixels"] == [ccp["w"], ccp["h"]]:
            warnings.warn(f"It seems the orientation = {ssk['orientation']} and the height and width are reversed")
            sensor.pixel_size = ssk["pixel_size"][::-1]   # reverse
            sensor.pixel_width, sensor.pixel_height = ssk["pixel_size"]
            sensor.calibration.cx, sensor.calibration.cy = ssk["photo_center_in_pixels"]
        else:
            raise NotImplementedError("Have no examples with sensor orientation != 1")

        # save calibration values, pix4d unit is pixel for this value
        sensor.calibration.software = self.software
        sensor.calibration.type = sensor.type
        sensor.calibration.f = sensor.focal_length * sensor.width / sensor.w_mm
        sensor.calibration.k1 = cicp["K1"]
        sensor.calibration.k2 = cicp["K2"]
        sensor.calibration.k3 = cicp["K3"]

        sensor.calibration.t1 = cicp["T1"]
        sensor.calibration.t2 = cicp["T2"]

        self.sensors[0] = sensor

        ####################
        # info for Photo() #
        ####################
        pmat = read_pmat(p4d_dict["param"]["pmat"])

        if raw_img_folder is not None:
            img_list = os.listdir(raw_img_folder)
            missing_photo = []
        else:
            img_list = []
            missing_photo = []

        for i, img_label in enumerate(pmat.keys()):
            # record the suffix of image
            img_label_no_suffix, img_suffix = os.path.splitext(img_label)
            if i == 0:
                self.photos._suffix = img_suffix

            img = idp.reconstruct.Photo(self.sensors[0])
            img.id = i
            img.label = img_label_no_suffix
            img.sensor_id = 0
            img.enabled = True

            # find raw image path
            if len(img_list) > 0:
                if img_label in img_list:
                    img_full_path = os.path.join(raw_img_folder, img_label)
                    img.path = idp.get_full_path(img_full_path)
                else:
                    missing_photo.append(img_label)

            # pix4d reverse calculation only need pmatrix
            img.transform = pmat[img_label]  # pmatrix

            # and just in case need this:
            img.cam_matrix = ccp[img_label]["cam_matrix"]  # K
            img.location = ccp[img_label]["cam_pos"]  # t
            img.rotation = ccp[img_label]["cam_rot"]   # R

            self.photos[i] = img

        if len(missing_photo) > 0:
            warnings.warn(
                f"Could not find {missing_photo} in given raw_img_folder"
                "[{raw_img_folder}]"
            )

        #####################
        # info for self.CRS #
        #####################
        self.crs = idp.shp.read_proj(p4d_dict["param"]["crs"])
        # record the default CRS, used for camera position obtain (and convertion)
        # check the def get_camera_position() function below
        self._proj_crs = idp.shp.read_proj(p4d_dict["param"]["crs"])

        ################################
        # info for outpus(PCD|DOM|DSM) #
        ################################
        if isinstance(p4d_dict["pcd"], (Path, str)) and Path(p4d_dict["pcd"]).exists():
            self.load_pcd(p4d_dict["pcd"])

        if isinstance(p4d_dict["dom"], (Path, str)) and Path(p4d_dict["dom"]).exists():
            self.load_dom(p4d_dict["dom"])

        if isinstance(p4d_dict["dsm"], (Path, str)) and Path(p4d_dict["dsm"]).exists():
            self.load_dsm(p4d_dict["dsm"])

    def load_pcd(self, pcd_path):
        """Manual load the point cloud file generated by this Pix4D project

        Parameters
        ----------
        pcd_path : str
            The path to point cloud file

        Caution
        -------
        The pix4d produced point cloud is offsetted (xyz-offset).
        This function already handle adding offset back to point cloud.
        If you need manual specify ``idp.PointCloud()`` by yourself, please do:

        .. code-block:: python

            >>> p4d = idp.Pix4D(project_path, param_folder)

            # wrong
            >>> pcd = idp.PointCloud(lotus_full_pcd)

            # correct
            >>> pcd = idp.PointCloud(lotus_full_pcd, offset=p4d.meta['p4d_offset'])
            # or 
            >>> pcd = idp.PointCloud(lotus_full_pcd, offset=p4d.offset_np)

        Example
        -------

        .. code-block:: python

            >>> import easyidp as idp
            >>> ... # define your project_path and param_folder path
            >>> p4d = idp.Pix4D(project_path, param_folder)
            >>> p4d.pcd
            None
            >>> p4d.load_pcd(pcd_path)
            >>> p4d.pcd
            <easyidp.PointCloud> object

        """
        if os.path.exists(pcd_path):
            self.pcd = idp.PointCloud(pcd_path, offset=self.meta["p4d_offset"])

    def load_dom(self, geotiff_path):
        """Manual load the DOM file generated by this Pix4D project

        Parameters
        ----------
        geotiff_path : str
            The path to DOM file

        Example
        -------

        .. code-block:: python

            >>> import easyidp as idp
            >>> ... # define your project_path and param_folder path
            >>> p4d = idp.Pix4D(project_path, param_folder)
            >>> p4d.dom
            None
            >>> p4d.load_dom(dom_path)
            >>> p4d.dom
            <easyidp.GeoTiff> object

        """
        if os.path.exists(geotiff_path):
            self.dom = idp.GeoTiff(geotiff_path)

    def load_dsm(self, geotiff_path):
        """Manual load the DSM file generated by this Pix4D project

        Parameters
        ----------
        geotiff_path : str
            The path to DSM file

        Example
        -------

        .. code-block:: python

            >>> import easyidp as idp
            >>> ... # define your project_path and param_folder path
            >>> p4d = idp.Pix4D(project_path, param_folder)
            >>> p4d.dsm
            None
            >>> p4d.load_dsm(dsm_path)
            >>> p4d.dsm
            <easyidp.GeoTiff> object

        """
        if os.path.exists(geotiff_path):
            self.dsm = idp.GeoTiff(geotiff_path)

    ################################
    # code for reverse calculation #
    ################################
    def _check_photo_type(self, photo):
        if isinstance(photo, str):
            if photo in self.photos.keys():
                return self.photos[photo]
            else:
                raise KeyError(
                    f"Could not find given image name [{photo}] in "
                    f"[{self.photos[0].label}, {self.photos[1].label}, ..., {self.photos[-1].label}]")
        elif isinstance(photo, idp.reconstruct.Photo):
            return photo
        else:
            raise TypeError("Only support <easyidp.Photo> or image_name string")

    def _external_internal_calc(self, points, photo, distort_correct=True):
        """Calculate backward projection by camera parameters, seems not correct, deprecated.

        Parameters
        ----------
        points : np.ndarray
            the nx3 xyz numpy matrix
        photo : str | easyidp.Photo object
            if str -> photo name / keys
        distort_correct : bool, optional
            whether calibrate lens distortion, by default True

        Returns
        -------
        coords_b : 2d ndarray
            'lower-left' coordiantes

        """
        photo = self._check_photo_type(photo)
        T = photo.location   # v1.0: T = param.img[image_name].cam_pos
        R = photo.rotation   # v1.0: R = param.img[image_name].cam_rot

        X_prime = (points - T).dot(R)
        xh = X_prime[:, 0] / X_prime[:, 2]
        yh = X_prime[:, 1] / X_prime[:, 2]

        # olderversion
        # f = param.F * param.img[image_name].w / param.w_mm
        # cx = param.Px * param.img[image_name].w / param.w_mm
        # cy = param.Py * param.img[image_name].h / param.h_mm
        calibration = self.sensors[photo.sensor_id].calibration

        if distort_correct:
            xb, yb = calibration.calibrate(xh, yh)
        else:
            f = calibration.f
            cx = calibration.cx
            cy = calibration.cy

            xb = f * xh + cx
            yb = f * yh + cy

        #xa = xb
        #ya = param.img[image_name].h - yb
        #coords_a = np.hstack([xa[:, np.newaxis], ya[:, np.newaxis]])
        #coords_b = np.hstack([xb[:, np.newaxis], yb[:, np.newaxis]])
        return np.vstack([xb, yb]).T


    def _pmatrix_calc(self, points, photo, distort_correct=True):
        """Calculate backward projection by pix4d pmatrix

        Parameters
        ----------
        points : np.ndarray
            the nx3 xyz numpy matrix
        photo : str | easyidp.Photo object
            if str -> photo name / keys
        distort_correct : bool, optional
            whether calibrate lens distortion, by default True

        Returns
        -------
        coords_b : 2d ndarray, 
            'lower-left' coordiantes
        """
        photo = self._check_photo_type(photo)

        xyz1_prime = np.insert(points, 3, 1, axis=1)
        xyz = (xyz1_prime).dot(photo.transform.T)  # v1.0: param.img[image_name].pmat.T
        u = xyz[:, 0] / xyz[:, 2]   # already the pixel coords
        v = xyz[:, 1] / xyz[:, 2]

        calibration = self.sensors[photo.sensor_id].calibration
        if distort_correct:
            xh, yh = calibration.calibrate(u, v)
            coords_b = np.vstack([xh, yh]).T
        else:
            coords_b = np.vstack([u, v]).T

        return coords_b


    def back2raw_crs(self, points_xyz, distort_correct=True, ignore=None, log=False):
        """Projects one GIS coordintates ROI (polygon) to all images

        Parameters
        ----------
        points_hv : ndarray (nx3)
            The 3D coordinates of polygon vertexm, in CRS coordinates
        distortion_correct : bool, optional
            Whether do distortion correction, by default True (back to raw image);
            If back to software corrected images without len distortion, set it to True. 
            Pix4D support do this operation, seems metashape not supported yet.
        ignore : str | None, optional
            Whether tolerate small parts outside image, check 
            :func:`easyidp.reconstruct.Sensor.in_img_boundary` for more details.

            - ``None``: strickly in image area;
            - ``x``: only y (vertical) in image area, x can outside image;
            - ``y``: only x (horizontal) in image area, y can outside image.

        log : bool, optional
            whether print log for debugging, by default False

        Returns
        -------
        dict,
            a dictionary like ``{img_name: pixel coordinate, ... }``

        Example
        -------

        Data preparation

        .. code-block:: python

            >>> import easyidp as idp

            >>> p4d = idp.Pix4D(
            ...     test_data.pix4d.lotus_folder,
            ...     raw_img_folder=test_data.pix4d.lotus_photos,
            ...     param_folder=test_data.pix4d.lotus_param
            ... )

            >>> plot =  np.array([   # N1E1 plot geo coordinate
            ...     [ 368020.2974959 , 3955511.61264302,      97.56272272],
            ...     [ 368022.24288365, 3955512.02973983,      97.56272272],
            ...     [ 368022.65361232, 3955510.07798313,      97.56272272],
            ...     [ 368020.69867274, 3955509.66725421,      97.56272272],
            ...     [ 368020.2974959 , 3955511.61264302,      97.56272272]
            ... ])

        Then use this function to find the previous ROI positions on the raw images:

        .. code-block:: python

            >>> out_dict = p4d.back2raw_crs(plot, distort_correct=True)

            >>> out_dict["DJI_0177"]
            array([[ 137.10982937, 2359.55887614],
                   [ 133.56116243, 2107.13954299],
                   [ 384.767746  , 2097.05639105],
                   [ 388.10993307, 2350.41225998],
                   [ 137.10982937, 2359.55887614]])

        """
        out_dict = {}
        sensor = self.sensors[0]
        # seems all teh calculation is based on no offset coordinate
        points_xyz = points_xyz - self.meta["p4d_offset"]

        if log:
            print(f'[Calculator][Judge]camera_name photo.width photo.height -> x.min \t x.max \t y.min \t y.max')

        for photo_name, photo in self.photos.items():
            if log:
                print(f'[Calculator][Judge]{photo.label} w:{photo.sensor.width} h:{photo.sensor.height} -> ', end='')
            #if method == 'exin':
            #    projected_coords = self._external_internal_calc(points, photo, distort_correct)
            projected_coords = self._pmatrix_calc(points_xyz, photo, distort_correct)
            coords = sensor.in_img_boundary(projected_coords, ignore=ignore, log=log)
            if coords is not None:
                out_dict[photo.label] = coords

        return out_dict


    def back2raw(self, roi, save_folder=None, **kwargs):
        """Projects several GIS coordintates ROIs (polygons) to all images

        Parameters
        ----------
        roi : easyidp.ROI | dict
            the <ROI> object created by easyidp.ROI() or dictionary
        save_folder : str, optional
            the folder to save projected preview images and json files, by default ""
        distortion_correct : bool, optional
            Whether do distortion correction, by default True (back to raw image);
            If back to software corrected images without len distortion, set it to True. 
            Pix4D support do this operation, seems metashape not supported yet.
        ignore : str | None, optional
            Whether tolerate small parts outside image, check 
            :func:`easyidp.reconstruct.Sensor.in_img_boundary` for more details.

            - ``None``: strickly in image area;
            - ``x``: only y (vertical) in image area, x can outside image;
            - ``y``: only x (horizontal) in image area, y can outside image.

        log : bool, optional
            whether print log for debugging, by default False

        Example
        -------

        Data prepare

        .. code-block:: python

            >>> import easyidp as idp

            >>> lotus = idp.data.Lotus()

            >>> p4d = idp.Pix4D(project_path=lotus.pix4d.project, 
                                raw_img_folder=lotus.photo,
                                param_folder=lotus.pix4d.param)

            >>> roi = idp.ROI(lotus.shp, name_field=0)
            [shp][proj] Use projection [WGS 84] for loaded shapefile [plots.shp]
            Read shapefile [plots.shp]: 100%|███████████████| 112/112 [00:00<00:00, 2481.77it/s]
            >>> roi = roi[0:2]

            >>> roi.get_z_from_dsm(lotus.pix4d.dsm)


        Then using this function to do backward projection:

        .. code-block:: python

            >>> out_all = p4d.back2raw(roi)
            {
                'N1W1': {
                    'DJI_0479': 
                        array([[  52.9824393 , 1253.05643133],
                               [  79.20465849,  979.90831093],
                               [ 363.27656888, 1000.07501881],
                               [ 337.25115499, 1273.15285336],
                               [  52.9824393 , 1253.05643133]]), 
                    'DJI_0480':
                        array([...])
                    ...
                }
                'N1W2': {...}   # output of `back2raw_crs()`
            }

        """
        out_dict = {}

        pbar = tqdm(roi.items(), desc=f"Backward roi to raw images")
        for k, points_xyz in pbar:
            if isinstance(save_folder, str) and os.path.isdir(save_folder):
                save_path = os.path.join(save_folder, k)
            else:
                save_path = None

            if points_xyz.shape[1] != 3:
                raise ValueError(
                    f"The back2raw function requires 3D roi with shape=(n, 3)"
                    f", but [{k}] is {points_xyz.shape}")

            one_roi_dict= self.back2raw_crs(points_xyz, **kwargs)

            out_dict[k] = one_roi_dict

        if save_folder is not None:
            idp.reconstruct.save_back2raw_json_and_png(self, out_dict, save_folder)

        return out_dict

    def get_photo_position(self, to_crs=None, refresh=False):
        """Get all photos' center geo position (on given CRS)

        Parameters
        ----------
        to_crs : pyproj.CRS, optional
            Transformed to another geo coordinate, by default None, the project.crs
        refresh : bool, optional
            
            - ``False`` : Use cached results (if have), by default
            - ``True`` : recalculate the photo position

        Returns
        -------
        dict
            The dictionary contains "photo.label": [x, y, z] coordinates

        Example
        -------

        Data prepare

        .. code-block:: python

            >>> import easyidp as idp

            >>> lotus = idp.data.Lotus()
            >>> p4d = idp.Pix4D(lotus.pix4d.project,lotus.photo, lotus.pix4d.param)

        Then use this function to get the photo position in 3D world:

        .. code-block:: python

            >>> out = p4d.get_photo_position()
            {
                'DJI_0422.JPG': array([ 368016.23334752, 3955491.97729229,     138.25541246]), 
                'DJI_0423.JPG': array([ 368016.33261375, 3955492.15845851,     138.05762001]),
                ...
            }

        """
        if self._photo_position_cache is not None and not refresh:
            return self._photo_position_cache.copy()
        else:
            out = {}
            pbar = tqdm(self.photos, desc=f"Getting photo positions")
            for p in pbar:
                if p.enabled:
                    # it has the same crs of DOM/DSM, project_default_crs, the pix4d feature
                    # it is different with metashape project logic, check Metashape.get_photo_position for more info
                    pos = p.location + self.meta["p4d_offset"]

                    # if specify the output crs: (to_crs is not None)
                    if isinstance(to_crs, pyproj.CRS):
                        if not self._proj_crs.equals(to_crs):  # need do proj convertion
                            pos = idp.geotools.convert_proj3d(pos, self._proj_crs, to_crs)
                    # not specify the output crs, then check the self.crs is the same with project default crs
                    else:
                        if not self._proj_crs.equals(self.crs):  # need do proj convertion
                            pos = idp.geotools.convert_proj3d(pos, self._proj_crs, self.crs)

                    out[p.label] = pos
                    p.position = pos

            self._photo_position_cache = out

            return out

    def sort_img_by_distance(self, img_dict_all, roi, distance_thresh=None, num=None, save_folder=None):
        """Advanced wrapper of sorting back2raw img_dict results by distance from photo to roi

        Parameters
        ----------
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
            the same structure as output of roi.back2raw(...)


        Example
        -------

        In the previous :func:`back2raw` results :

        .. code-block:: python

            >>> out_all = p4d.back2raw(roi)
            {
                'N1W1': {
                    'DJI_0479': 
                        array([[  52.9824393 , 1253.05643133],
                               [  79.20465849,  979.90831093],
                               [ 363.27656888, 1000.07501881],
                               [ 337.25115499, 1273.15285336],
                               [  52.9824393 , 1253.05643133]]), 
                    'DJI_0480':
                        array([...])
                    ...
                }
                'N1W2': {...}   # output of `back2raw_crs()`
            }

        The image are in chaos order, in most application cases, probable only 1-3 closest images 
        (to ROI in real world) are required, so this function is provided to sort/filter out.

        In the following example, it filtered 3 images whose distance from camera to ROI in real 
        world smaller than 10m:

        .. code-block:: python

            >>> img_dict_sort = p4d.sort_img_by_distance(
            ...     out_all, roi,
            ...     distance_thresh=10,  # distance threshold is 10m
            ...     num=3   # only keep 3 closest images
            ... )

            >>> img_dict_sort
            {
                'N1W1': {
                    'DJI_0500': array([[1931.09279469, 2191.59919979],
                                       [1939.92139124, 1930.65101348],
                                       [2199.9439422 , 1939.32128527],
                                       [2191.19230849, 2200.557026  ],
                                       [1931.09279469, 2191.59919979]]), 
                    'DJI_0517': array([[2870.94915401, 2143.3570243 ],
                                       [2596.8790503 , 2161.04730612],
                                       [2578.87033498, 1886.89058023],
                                       [2853.13891851, 1869.99769984],
                                       [2870.94915401, 2143.3570243 ]]), 
                    'DJI_0518': array([[3129.43264924, 1984.91814896],
                                       [2856.71879306, 2002.03817639],
                                       [2838.71418138, 1730.00287388],
                                       [3111.73360179, 1713.76233134],
                                       [3129.43264924, 1984.91814896]])
                }, 
                'N1W2': {
                    'DJI_0500': array([[2214.36789052, 2200.35979344],
                                       [2221.8996575 , 1940.70687713],
                                       [2479.9825464 , 1949.3909589 ],
                                       [2472.52171907, 2209.40355333],
                                       [2214.36789052, 2200.35979344]]), 
                    'DJI_0517': array([[2849.82108263, 1845.6733702 ],
                                       [2577.37309441, 1863.60741328],
                                       [2559.80046778, 1592.07656949],
                                       [2832.52942622, 1574.92640413],
                                       [2849.82108263, 1845.6733702 ]]), 
                    'DJI_0516': array([[2891.61686486, 2542.98979632],
                                       [2616.06780032, 2559.41601014],
                                       [2598.43900454, 2282.36641612],
                                       [2874.23023492, 2266.71552931],
                                       [2891.61686486, 2542.98979632]])
                }
            }

        Or pick the closest one image:

        .. code-block:: python

            >>> img_dict_sort = p4d.sort_img_by_distance(
            ...     out_all, roi,
            ...     num=1   # only keep 1 closest images
            ... )

            >>> img_dict_sort
            {
                'N1W1': {
                    'DJI_0500': array([[1931.09279469, 2191.59919979],
                                       [1939.92139124, 1930.65101348],
                                       [2199.9439422 , 1939.32128527],
                                       [2191.19230849, 2200.557026  ],
                                       [1931.09279469, 2191.59919979]])
                }, 
                'N1W2': {
                    'DJI_0500': array([[2214.36789052, 2200.35979344],
                                       [2221.8996575 , 1940.70687713],
                                       [2479.9825464 , 1949.3909589 ],
                                       [2472.52171907, 2209.40355333],
                                       [2214.36789052, 2200.35979344]])
                }
            }

        You can use ``list(dict.keys())[0]`` to get the image name automatically to iterate each plot:

        .. code-block:: python

            for plot_name, plot_value in img_dict_sort.items():
                img_name = list(plot_value.key())[0]

                coord = plot_value[img_name]
                # or
                coord = img_dict_sort[plot_name][img_name]

        See also
        --------
        easyidp.reconstruct.sort_img_by_distance

        """
        return idp.reconstruct.sort_img_by_distance(self, img_dict_all, roi, distance_thresh, num, save_folder)
    
    def show_roi_on_img(self, img_dict, roi_name, img_name=None, **kwargs):
        """Visualize the specific backward projection results for given roi on the given image.

        Parameters
        ----------
        img_dict : dict
            The backward results from back2raw()
        roi_name : str
            The roi name to show
        img_name : str
            the image file name. by default None, plotting all available images
        corrected_poly_coord : np.ndarray, optional
            the corrected 2D polygon pixel coordiante on the image (if have), by default None
        title : str, optional
            The image title displayed on the top, by default None -> ``ROI [roi_name] on [img_name]``
        save_as : str, optional
            file path to save the output figure, by default None
        show : bool, optional
            whether display (in jupyter notebook) or popup (in command line) the figure, by default False
        color : str, optional
            the polygon line color, by default 'red'
        alpha : float, optional
            the polygon transparency, by default 0.5
        dpi : int, optional
            the dpi of produced figure, by default 72

        Example
        -------

        .. code-block:: python

            >>> img_dict_ms = roi.back2raw(ms)

        Check the "N1W1" ROI on image "DJI_0479.JPG":

        .. code-block:: python

            >>> ms.show_roi_on_img(img_dict_ms, "N1W1", "DJI_0479")
            
        Check the "N1W1" ROI on all available images:

            >>> ms.show_roi_on_img(img_dict_ms, "N1W1")

        For more details, please check in :ref:`this example <show-one-roi-on-img-demo>`

        See also
        --------
        easyidp.visualize.draw_polygon_on_img, easyidp.visualize.draw_backward_one_roi
        """
        # check if given has values
        if roi_name not in img_dict.keys() or \
                (img_name is not None and \
                    img_name not in img_dict[roi_name].keys()):
            raise IndexError(f"Could not find backward results of plot [{roi_name}] on image [{img_name}]")
        
        if img_name is not None and img_name not in self.photos.keys():
            raise FileNotFoundError(f"Could not find the image file [{img_name}] in the Pix4D project")
        
        if 'title' not in kwargs:
            kwargs['title'] = f"ROI [{roi_name}] on [{img_name}]"

        if 'show' not in kwargs:
            kwargs['show'] = True
    
        if img_name is not None:
            idp.visualize.draw_polygon_on_img(
                img_name, 
                img_path=self.photos[img_name].path, 
                poly_coord=img_dict[roi_name][img_name], 
                **kwargs
                )
        else:
            idp.visualize.draw_backward_one_roi(
                self, img_dict[roi_name], 
                **kwargs
            )

####################
# code for file IO #
####################

def _match_suffix(folder, ext):
    """
    find the *first* file by given suffix, e.g. *.tif
    Parameters
    ----------
    folder: str
        the path of folder want to find
    ext: str | list
        str -> e.g. "tif"
        list -> e.g. ["ply", "laz"]
            will find all suffix files

    Returns
    -------
    find_path: str | None
        if find -> str of path
        if not find -> None
    """
    find_path = None
    if Path(folder).exists():
        if isinstance(ext, str):
            one_ext = True
        else:
            one_ext = False

        if one_ext:
            for file in os.listdir(folder):
                if file.endswith(ext):
                    find_path = f"{folder}/{file}"
                    return find_path
        else:
            for ex in ext:
                for file in os.listdir(folder):
                    if file.endswith(ex):
                        find_path = f"{folder}/{file}"
                        return find_path

    return find_path


def parse_p4d_param_folder(param_path:str):
    """Get full file path of parameter folder (``...\\project_name\\1_initial\\params.``) of Pix4D project.

    Parameters
    ----------
    param_path : str
        The param folder path of pix4d project.

    Returns
    -------
    dict
        The dictionary contains pix4d params, ``keys=["project_name", "xyz", "pmat", "cicp", "ccp", "campos", "ssk", "crs"]``

    Note
    ----

    We use the following parameters [1]_:

    - ``project_name`` : the project name
    - ``xyz``: the full file path of ``*_offset.xyz``, contains the point cloud offset values
    - ``pmat``: the full file path of ``*_pmatrix.txt`` file, contains the compressed internal and external camera parameters.
    - ``cicp``: the full file path of ``*_pix4d_calibrated_internal_camera_parameters.cam``, it contains information about the optimized (computed) internal camera parameters.
    - ``ccp``: the full file path of ``*_calibrated_camera_parameters.txt``, it contains the information of each calibrated camera.
    - ``campos``: the full file path of ``*_calibrated_images_position.txt``, the position information of each calibrated camera.
    - ``ssk``: the full file path of ``*_camera.ssk``, it contains information about the camera parameters.
    - ``crs`` : the full file path of ``*_wkt.prj``, it contains the projection of the output coordinate system in the projection format.


    Example
    -------

    Data prepare

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> param_folder = str(test_data.pix4d.maize_folder / "1_initial" / "params")
        '/Users/<user>/Library/Application Support/easyidp.data/data_for_tests/pix4d/maize_tanashi/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d/1_initial/params'

    Then use this function:

    .. code-block:: python

        >>> param = idp.pix4d.parse_p4d_param_folder(param_folder)

        >>> param.keys()
        dict_keys(['project_name', 'xyz', 'pmat', 'cicp', 'ccp', 'campos', 'ssk', 'crs'])

        >>> param['xyz']
        '/Users/<user>/Library/Application Support/easyidp.data/data_for_tests/pix4d/maize_tanashi/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d/1_initial/params/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_offset.xyz'

        >>> param['ccp']
        '/Users/<user>/Library/Application Support/easyidp.data/data_for_tests/pix4d/maize_tanashi/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d/1_initial/params/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_calibrated_camera_parameters.txt'

    References
    ----------
    .. [1] What does the Output Params Folder contain? https://support.pix4d.com/hc/en-us/articles/202977149-What-does-the-Output-Params-Folder-contain

    """
    param_dict = {}
    # keys = ["project_name", "xyz", "pmat", "cicp", "ccp"]

    param_files = os.listdir(param_path)

    if len(param_files) < 6:
        raise FileNotFoundError(
            f"Given param folder [{idp.get_full_path(param_path)}] "
            "does not have enough param files to parse"
        )

    project_name = os.path.commonprefix(param_files)
    # > "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_"
    if project_name[-1] == '_':
        project_name = project_name[:-1]
    param_dict["project_name"] = project_name

    xyz_file = f"{param_path}/{project_name}_offset.xyz"
    if os.path.exists(xyz_file):
        param_dict["xyz"] = xyz_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{xyz_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path."
        )

    pmat_file = f"{param_path}/{project_name}_pmatrix.txt"
    if os.path.exists(pmat_file):
        param_dict["pmat"] = pmat_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{pmat_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path."
        )

    cicp_file = f"{param_path}/{project_name}_pix4d_calibrated_internal_camera_parameters.cam"
    # two files with the same string
    # {project_name}_      calibrated_internal_camera_parameters.cam
    # {project_name}_pix4d_calibrated_internal_camera_parameters.cam
    if os.path.exists(cicp_file):
        param_dict["cicp"] = cicp_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{cicp_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path."
        )

    ccp_file = f"{param_path}/{project_name}_calibrated_camera_parameters.txt"
    if os.path.exists(ccp_file):
        param_dict["ccp"] = ccp_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{ccp_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path."
        )

    campos_file = f"{param_path}/{project_name}_calibrated_images_position.txt"
    if os.path.exists(campos_file):
        param_dict["campos"] = campos_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{campos_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path."
        )

    ssk_file = f"{param_path}/{project_name}_camera.ssk"
    if os.path.exists(ssk_file):
        param_dict["ssk"] = ssk_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{ssk_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path."
        )

    prj_file = f"{param_path}/{project_name}_wkt.prj"
    if os.path.exists(prj_file):
        param_dict["crs"] = prj_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{ssk_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path."
        )

    return param_dict


def parse_p4d_project(project_path:str, param_folder=None):
    """
    A fuction to automatically analyze related subfiles in pix4d project folder

    Parameters
    ----------
    project_path: str
        the path to pix4d project file, that folder should contains the following sub-folder:

        .. code-block:: text

            \\project_path
            |--- 1_initial\\
            |--- 2_densification\\
            |--- 3_dsm_ortho\\


    param_folder: str, default None
        | if not given, it will parse as a standard pix4d project, and trying
        |     to get the project name from ``1_initial/param`` folder
        | if it is not a standard pix4d project (re-orgainzed folder), need manual
        |     specify the path to param folder, in order to parse project_name
        |     for later usage.

    Returns
    -------
    p4d: dict
        a python dictionary that contains the path to each file.
        
        .. code-block:: python

            {
                "project_name": the prefix of whole project file.
                "param": the folder of parameters
                "pcd": the point cloud file
                "dom": the digital orthomosaic file
                "dsm": the digital surface model file
                "undist_raw": the undistorted images corrected by the pix4d software (when original image unable to find)
            }

    Notes
    -----
    Project_name can be extracted from parameter folder prefix in easyidp 2.0, no need manual specify.
    To find the outputs, it will pick the first file that fits the expected file format.

    Example
    -------

    Data prepare

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> test_folder = test_data.pix4d.maize_folder
        PosixPath('/Users/<user>/Library/Application Support/easyidp.data/data_for_tests/pix4d/maize_tanashi/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d')
        >>> proj_name1 = "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d"

    Then use this function to parse given pix4d:

    .. code-block:: python

        >>> p4d_p = idp.pix4d.parse_p4d_project(test_folder)
        >>> p4d_p.keys()
        dict_keys(['param', 'pcd', 'dom', 'dsm', 'undist_raw', 'project_name'])

        >>> p4d_p['project_name']
        'maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d'

        >>> p4d_p['dom']
        PosixPath('/Users/<user>/Library/Application Support/easyidp.data/data_for_tests/pix4d/maize_tanashi/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d/3_dsm_ortho/2_mosaic/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_transparent_mosaic_group1.tif')


    """
    p4d = {"param": None, "pcd": None, "dom": None, "dsm": None, "undist_raw": None, "project_name": None}

    project_path = idp.get_full_path(project_path)
    sub_folder = os.listdir(project_path)

    #################################
    # parse defalt 1_initial folder #
    #################################
    if param_folder is None:
        param_folder = project_path / "1_initial" / "params"

    undist_folder = project_path / "1_initial" / "images" / "undistorted_images"

    # check whether a correct pix4d project folder
    if '1_initial' not in sub_folder and param_folder is None:
        raise FileNotFoundError(f"Current folder [{project_path}] is not a standard pix4d projects folder, please manual speccify `param_folder`")

    if os.path.exists(param_folder) and len(os.listdir(param_folder)) > 0:
        param = parse_p4d_param_folder(param_folder)
        p4d["param"] = param
        project_name = param["project_name"]
        p4d["project_name"] = param["project_name"]
    else:
        raise FileNotFoundError(
            "Can not find pix4d parameter in given project folder"
        )

    if os.path.exists(undist_folder):
        p4d["undist_raw"] = undist_folder

    ######################
    # parse output files #
    ######################

    # point cloud file
    pcd_folder = project_path / "2_densification" / "point_cloud"
    ply_file   = pcd_folder / f"{project_name}_group1_densified_point_cloud.ply"
    laz_file   = pcd_folder / f"{project_name}_group1_densified_point_cloud.laz"
    las_file   = pcd_folder / f"{project_name}_group1_densified_point_cloud.las"

    if pcd_folder.exists():
        if ply_file.exists():
            p4d["pcd"] = ply_file
        elif las_file.exists():
            p4d["pcd"] = las_file
        elif laz_file.exists():
            p4d["pcd"] = laz_file
        else:
            force = _match_suffix(pcd_folder, ["ply", "las", "laz"])
            if force is not None:
                p4d["pcd"] = force
            else:
                warnings.warn(
                    f"Unable to find any point cloud output file "
                    "[*.ply, *.las, *.laz] in the project folder "
                    "[{pcd_folder}]. Please specify manually."
                )

    # digital surface model DSM file
    dsm_folder = project_path / "3_dsm_ortho" / "1_dsm"
    dsm_file = dsm_folder / f"{project_name}_dsm.tif"

    if dsm_folder.exists():
        if dsm_file.exists():
            p4d["dsm"] = dsm_file
        else:
            force = _match_suffix(dsm_folder, "tif")
            if force is not None:
                p4d["dsm"] = force
            else:
                warnings.warn(
                    f"Unable to find any DSM output file "
                    "[*.ply, *.las, *.laz] in the project folder "
                    "[{dense_folder}]. Please specify manually."
                )

    dom_folder = project_path / "3_dsm_ortho" / "2_mosaic"
    dom_file = dom_folder / f"{project_name}_transparent_mosaic_group1.tif"
    if dom_folder.exists():
        if dom_file.exists():
            p4d["dom"] = dom_file
        else:
            force = _match_suffix(dom_folder, "tif")
            if force is not None:
                p4d["dom"] = force
            else:
                warnings.warn(
                    f"Unable to find any DOM output file "
                    "[*.ply, *.las, *.laz] in the project folder "
                    "[{dense_folder}]. Please specify manually."
                )

    return p4d


def read_xyz(xyz_path):
    """read pix4d file ``*_offset.xyz``

    Parameters
    ----------
    xyz_path: str
        the path to target offset.xyz file

    Returns
    -------
    x, y, z: float

    Note
    ----

    The offset.xyz file looks like:

    .. code-block:: text

        368009.000 3955854.000 97.000

    Example
    -------

    Data prepare

    .. code-block:: python

        >>> import numpy as np
        >>> np.set_printoptions(suppress=True)

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> param_folder = str(test_data.pix4d.maize_folder / "1_initial" / "params")
        >>> param = idp.pix4d.parse_p4d_param_folder(param_folder)

    Then use this function:

    .. code-block:: python

        >>> idp.pix4d.read_xyz(param['xyz'])
        array([ 368009., 3955854.,      97.])

    """
    with open(xyz_path, 'r') as f:
        x, y, z = f.read().split(' ')
    return np.array([float(x), float(y), float(z)])


def read_pmat(pmat_path):
    """read pix4d file ``*_pmatrix.txt``

    Parameters
    ----------
    pmat_path: str
        the path of pmatrix file.type

    Returns
    -------
    dict

        .. code-block:: python

            pmat_dict = {
                "DJI_0000.JPG": nparray(3x4), 
                ... ,
                "DJI_9999.JPG": nparray(3x4)
            }

    Note
    ----

    The pmatrix.txt file looks like:

    .. code-block:: python

        DJI_0954.JPG 3111.599161 -2366.736021 -2308.589802 -65840.192261 -2444.444098 -3031.331712 -1800.672677 18549.006987 0.005818 0.038647 -0.999236 31.851547 
        DJI_0955.JPG 2962.002548 -2547.748788 -2312.702452 -69244.748198 -2620.207260 -2895.573136 -1776.758596 27157.887214 0.003220 0.034307 -0.999406 31.293337
        DJI_0956.JPG 3756.991075 -1038.548434 -2327.984283 -49670.339047 -1138.894683 -3738.320518 -1770.474397 -22001.847467 -0.015923 0.031981 -0.999362 31.327806
        DJI_0957.JPG 3923.542113 562.037059 -2214.273525 -8825.859811 514.572422 -3880.691126 -1755.606046 -68101.592371 0.001783 0.022713 -0.999740 30.301800
        ...

    Example
    -------

    Data prepare

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> param_folder = str(test_data.pix4d.maize_folder / "1_initial" / "params")
        >>> param = idp.pix4d.parse_p4d_param_folder(param_folder)

    Then use this function:

    .. code-block:: python

        >>> idp.pix4d.read_pmat(param['pmat'])
        {
            'DJI_0954.JPG': 
                array([[  3111.599161,  -2366.736021,  -2308.589802, -65840.192261],
                    [ -2444.444098,  -3031.331712,  -1800.672677,  18549.006987],
                    [     0.005818,      0.038647,     -0.999236,     31.851547]]), 
            'DJI_0955.JPG':
                array([...])
            ...
        }

    """
    pmat_nb = np.loadtxt(pmat_path, dtype=float, delimiter=None, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,))
    pmat_names = np.loadtxt(pmat_path, dtype=str, delimiter=None, usecols=0)

    pmat_dict = {}

    for i, name in enumerate(pmat_names):
        pmat_dict[name] = pmat_nb[i, :].reshape((3, 4))

    return pmat_dict


def read_cicp(cicp_path):
    """Read ``*_pix4d_calibrated_internal_camera_parameters.cam`` for :class:`easyidp.reconstruct.Sensor` object

    Parameters
    ----------
    cicp_path: str
        file path

    Returns
    -------
    dict

        .. code-block:: python

            cicp_dict.keys() = 
                ['F', 'Px', 'Py', 'K1', 'K2', 'K3', 'T1', 'T2', 'w_mm', 'h_mm']

    Notes
    -----
    It is the info about sensor, the file looks like:

    .. code-block:: text

        Pix4D camera calibration file 0
        #Focal Length mm assuming a sensor width of 17.49998592000000030566x13.12498944000000200560mm
        F 15.01175404934517487732
        #Principal Point mm
        Px 8.48210511970419922534
        Py 6.33434629978042273990
        #Symmetrical Lens Distortion Coeffs
        K1 0.03833474118270804865
        K2 -0.01750917966495743258
        K3 0.02049798716391852335
        #Tangential Lens Distortion Coeffs
        T1 0.00240851666319534747
        T2 0.00292562392135245920

    Example
    -------

    Data prepare

    .. code-block:: python

        >>> import numpy as np
        >>> np.set_printoptions(suppress=True)

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> param_folder = str(test_data.pix4d.maize_folder / "1_initial" / "params")
        >>> param = idp.pix4d.parse_p4d_param_folder(param_folder)

    Then use this function:

    .. code-block:: python

        >>> idp.pix4d.read_cicp(param['cicp'])
        {
            'w_mm': 17.49998592, 
            'h_mm': 13.124989440000002, 
            'F': 15.011754049345175, 
            'Px': 8.4821051197042, 
            'Py': 6.334346299780423, 
            'K1': 0.03833474118270805, 
            'K2': -0.017509179664957433, 
            'K3': 0.020497987163918523, 
            'T1': 0.0024085166631953475, 
            'T2': 0.002925623921352459
        }

    """
    with open(cicp_path, 'r') as f:
        key_pool = ['F', 'Px', 'Py', 'K1', 'K2', 'K3', 'T1', 'T2']
        cam_dict = {}
        for line in f.readlines():
            sp_list = line.split(' ')
            if len(sp_list) == 2:  # lines with param.name in front
                lead, contents = sp_list[0], sp_list[1]
                if lead in key_pool:
                    cam_dict[lead] = float(contents[:-1])
            elif len(sp_list) == 9:
                # one example:
                # > Focal Length mm assuming a sensor width of 12.82x8.55mm\n
                w_h = sp_list[8].split('x')
                cam_dict['w_mm'] = float(w_h[0])  # extract e.g. 12.82
                cam_dict['h_mm'] = float(w_h[1][:-4])  # extract e.g. 8.55
    return cam_dict


def read_ccp(ccp_path):
    """Read ``*_calibrated_camera_parameters.txt`` for :class:`easyidp.reconstruct.Photo` object

    Parameters
    ----------
    ccp_path: str
        file path

    Returns
    -------
    dict

        .. code-block:: python

            img_configs = {
                'w': 4608, 
                'h': 3456, 
                'Image1.JPG': {
                    'cam_matrix':  array([[...]]), 
                    'rad_distort': array([ 0.03833474, ...]),
                    'tan_distort': array([0.00240852, ...]), 
                    'cam_pos':     array([ 21.54872207, ...]), 
                    'cam_rot':     array([[ 0.78389904, ...]])}, 
                    
                'Image2.JPG':
                    {...}
                }

    Notes
    -----
    It is the camera position info in local coordinate, the file looks like:

    .. code-block:: text

        fileName imageWidth imageHeight
        camera matrix K [3x3]
        radial distortion [3x1]
        tangential distortion [2x1]
        camera position t [3x1]
        camera rotation R [3x3]
        camera model m = K [R|-Rt] X

        DJI_0954.JPG 4608 3456
        3952.81247514184087776812 0 2233.46124792750424603582
        0 3952.81247514184087776812 1667.92521335858214115433
        0 0 1
        0.03833474118270804865 -0.01750917966495743258 0.02049798716391852335
        0.00240851666319534747 0.00292562392135245920
        21.54872206687879199194 -29.58734160676452162875 30.        85702810138878149360
        0.78389904231994589345 -0.62058396220897726892 -0.      01943803742353054573
        -0.62086105345046738169 -0.78318706257080084043 -0.     03390541741516269608
        0.00581753884797473961 0.03864674463298682638 -0.99923600083815289352

        DJI_0955.JPG ...

    Example
    -------

    Data prepare

    .. code-block:: python

        >>> import numpy as np
        >>> np.set_printoptions(suppress=True)

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> param_folder = str(test_data.pix4d.maize_folder / "1_initial" / "params")
        >>> param = idp.pix4d.parse_p4d_param_folder(param_folder)

    Then use this function:

    .. code-block:: python

        >>> ccp = idp.pix4d.read_ccp(param['ccp'])
        >>> ccp.keys()
        dict_keys(['DJI_0954.JPG', 'w', 'h', 'DJI_0955.JPG', ... , 'DJI_0091.JPG', 'DJI_0092.JPG'])

        >>> ccp['w']
        4608

        >>> ccp['DJI_0954.JPG']
        {
            'cam_matrix': 
                array([[3952.81247514,    0.        , 2233.46124793],
                       [   0.        , 3952.81247514, 1667.92521336],
                       [   0.        ,    0.        ,    1.        ]]), 
                       
            'rad_distort': 
                array([ 0.03833474, -0.01750918,  0.02049799]),

            'tan_distort': 
                array([0.00240852, 0.00292562]), 

            'cam_pos': 
                array([ 21.54872207, -29.58734161,  30.8570281 ]), 

            'cam_rot': 
                array([[ 0.78389904, -0.62058396, -0.01943804],
                       [-0.62086105, -0.78318706, -0.03390542],
                       [ 0.00581754,  0.03864674, -0.999236  ]])
        }

    """
    with open(ccp_path, 'r') as f:
        '''
        # for each block
        1   fileName imageWidth imageHeight 
        2-4 camera matrix K [3x3]
        5   radial distortion [3x1]
        6   tangential distortion [2x1]
        7   camera position t [3x1]
        8-10   camera rotation R [3x3]
        '''
        lines = f.readlines()

    img_configs = {}

    file_name = ""
    cam_mat_line1 = ""
    cam_mat_line2 = ""
    cam_rot_line1 = ""
    cam_rot_line2 = ""
    for i, line in enumerate(lines):
        if i < 8:
            pass
        else:
            block_id = (i - 7) % 10
            if block_id == 1:  # [line]: fileName imageWidth imageHeight
                file_name, w, h = line[:-1].split()  # ignore \n character
                img_configs[file_name] = {}
                img_configs['w'] = int(w)
                img_configs['h'] = int(h)
            elif block_id == 2:
                cam_mat_line1 = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 3:
                cam_mat_line2 = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 4:
                cam_mat_line3 = np.fromstring(line, dtype=float, sep=' ')
                img_configs[file_name]['cam_matrix'] = np.vstack([cam_mat_line1, cam_mat_line2, cam_mat_line3])
            elif block_id == 5:
                img_configs[file_name]['rad_distort'] = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 6:
                img_configs[file_name]['tan_distort'] = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 7:
                img_configs[file_name]['cam_pos'] = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 8:
                cam_rot_line1 = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 9:
                cam_rot_line2 = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 0:
                cam_rot_line3 = np.fromstring(line, dtype=float, sep=' ')
                cam_rot = np.vstack([cam_rot_line1, cam_rot_line2, cam_rot_line3])
                img_configs[file_name]['cam_rot'] = cam_rot

    return img_configs


def read_campos_geo(campos_path):
    """Read ``*_calibrated_images_position.txt`` for :class:`easyidp.reconstruct.Photo.position` (geo_location)

    Parameters
    ----------
    campos_path : str
        file path

    Returns
    -------
    dict

        .. code-block:: python

            campos_dict = {
                "Image1.JPG": np.array([x, y ,z]), 
                "Image2.JPG": ...
                ...
            }

    Notes
    -----
    this file contains the geo position of each camera, and looks like:

    .. code-block:: text

        DJI_0954.JPG,368030.548722,3955824.412658,127.857028
        DJI_0955.JPG,368031.004387,3955824.824967,127.381322
        DJI_0956.JPG,368033.252520,3955826.479610,127.080709
        DJI_0957.JPG,368032.022104,3955826.060493,126.715974
        DJI_0958.JPG,368031.901165,3955826.109158,126.666393
        DJI_0959.JPG,368030.686490,3955830.981070,127.327741


    Example
    -------

    Data prepare

    .. code-block:: python

        >>> import numpy as np
        >>> np.set_printoptions(suppress=True)

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> param_folder = str(test_data.pix4d.maize_folder / "1_initial" / "params")
        >>> param = idp.pix4d.parse_p4d_param_folder(param_folder)

    Then use this function:

    .. code-block:: python

        >>> idp.pix4d.read_campos_geo(param['campos'])
        {
            'DJI_0954.JPG': 
                array([ 368030.548722, 3955824.412658,     127.857028]), 

            'DJI_0955.JPG': 
                array([ 368031.004387, 3955824.824967,     127.381322]),

            ...
        }

    """
    with open(campos_path, 'r') as f:
        cam_dict = {}
        for line in f.readlines():
            sp_list = line.split(',')
            if len(sp_list) == 4: 
                cam_dict[sp_list[0]] = np.array(sp_list[1:], dtype=float)

    return cam_dict


def read_cam_ssk(ssk_path):
    """Get the camera model name, for :class:`easyidp.reconstruct.Sensor` object

    Parameters
    ----------
    ssk_path : str

    Returns
    -------
    dict
    
        .. code-block:: python

            ssk_info = 
            {
                "label":                  str
                "type":                   str, e.g. frame / fisheye ...
                "pixel_size":             [h, w]
                "pixel_size_unit":        "mm"
                "image_size_in_pixels":   [h ,w]
                "orientation":            1   # guess 0 -> w, h?
                "photo_center_in_pixels": [h, w]
            }

    Note
    ----
    SSK file contents:

    .. code-block:: text

        begin camera_parameters FC550_DJIMFT15mmF1.7ASPH_15.0_4608x3456 (RGB)(1)
            focal_length:                      15.00522620622411729130
            ppac:                              0.02793232590500918308 -0.02181191393910364776
            ppbs:                               0 0
            film_format:                       13.12498944000000200560 17.49998592000000030566
            lens_distortion_flag:              off
            io_required:                       yes
            camera_type:                       frame
            media_type:                        digital
            pixel_size:                        3.79774000000000011568 3.79774000000000011568
            image_size_in_pixels:              3456 4608
            scanline_orientation:              4
            photo_coord_sys_orientation:       1
            photo_coord_sys_origin:            1727.50000000000000000000 2303.50000000000000000000
            focal_length_calibration_flag:     off
            calibrated_focal_length_stddev:    0.03
            ppac_calibration_flag:             off
            calibrated_ppac_stddevs:           0.003   0.003
            self_calibration_enabled_params:   0
            antenna_offsets:                   0   0   0
        end camera_parameters

    Example
    -------

    Data prepare

    .. code-block:: python

        >>> import numpy as np
        >>> np.set_printoptions(suppress=True)

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> param_folder = str(test_data.pix4d.maize_folder / "1_initial" / "params")
        >>> param = idp.pix4d.parse_p4d_param_folder(param_folder)

    Then use this function:

    .. code-block:: python

        >>> idp.pix4d.read_cam_ssk(param['ssk'])
        {
            'label': 'FC550_DJIMFT15mmF1.7ASPH_15.0_4608x3456', 
            'type': 'frame', 
            'pixel_size': [3.79774, 3.79774], 
            'image_size_in_pixels': [3456, 4608], 
            'orientation': 1, 
            'photo_center_in_pixels': [1727.5, 2303.5]
        }

    """
    with open(ssk_path, 'r') as f:
        ssk_info = {}
        for line in f.readlines():
            if "begin camera_parameters" in line:
                # > ['begin', 'camera_parameters', 'FC550_DJIMFT15mmF1.7ASPH_15.0_4608x3456', '(RGB)(1)']
                ssk_info["label"] = line.split(' ')[2]
            elif "camera_type" in line:
                # > ['', 'camera_type:', '', '',... '', '', 'frame\n']
                ssk_info["type"] = str(line.split(' ')[-1][:-1])  # last and rm \n
            elif "pixel_size" in line:
                # > ['', 'pixel_size:', '', ..., '', '3.79774000000000011568', '3.79774000000000011568']
                # the order is h, w, if orientation == 1
                # because: image_size_in_pixels: 3456 4608
                lsp = line.split(' ')
                ssk_info["pixel_size"] = [float(lsp[-2]), float(lsp[-1])]
            elif "image_size_in_pixels" in line:
                # double check with ccp imageWidth & imageHeight
                lsp = line.split(' ')
                ssk_info["image_size_in_pixels"] = [int(lsp[-2]), int(lsp[-1])]
            elif "photo_coord_sys_orientation" in line:
                ssk_info["orientation"] = int(line.split(' ')[-1])
            elif "photo_coord_sys_origin" in line:
                lsp = line.split(' ')
                ssk_info["photo_center_in_pixels"] = [float(lsp[-2]), float(lsp[-1])]

    return ssk_info