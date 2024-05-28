import os
import pyproj
import zipfile
import numpy as np
import warnings
from pathlib import Path
from tabulate import tabulate
from xml.etree import ElementTree
import xml.dom.minidom as minidom
from tqdm import tqdm
from copy import copy as ccopy

import easyidp as idp


class Metashape(idp.reconstruct.Recons):
    """the object for each chunk in Metashape 3D reconstruction project"""

    def __init__(self, project_path=None, chunk_id=None, raw_img_folder=None, check_img_existance=True):
        """The method to initialize the Metashape class

        Parameters
        ----------
        project_path : str, optional
            The metashape project file to open, like "xxxx.psx",, by default None, means create an empty class
        chunk_id : int or str, optional
            The chunk id or name(label) want to open, by default None, open the first chunk.
        raw_img_folder : str, optional
            the original UAV image folder, by default None
        check_img_existance : bool
            Ignore the missing photos when set to False, suitable for testing project with just a few images, to avoid the FileNotFoundError

        Example
        -------

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

        Then open the demo Metashape project:

        .. code-block:: python

            >>> ms = idp.Metashape(test_data.metashape.lotus_psx)
            <'Lotus.psx' easyidp.Metashape object with 1 active chunks>

              id  label
            ----  -------
            -> 0  Chunk 1

        Or you can create an empty class and then open project:

        .. code-block:: python

            >>> ms = idp.Metashape()
            >>> ms.open_project(test_data.metashape.lotus_psx)
            <'Lotus.psx' easyidp.Metashape object with 1 active chunks>

            id  label
            ----  -------
            -> 0  Chunk 1

        .. caution::

            One metashape project may have several chunks, and each ``easyidp.Metashape`` project could only handle with only one chunk at once. 

            The arrow before ID shows which chunk has been opened

            .. code-block:: text

                <'multichunk.psx' easyidp.Metashape object with 2 active chunks>

                  id  label
                ----  ------------
                -> 1  multiple_bbb
                   2  miltiple_aaa

        """
        super().__init__()
        self.software = "metashape"
        self.transform = idp.reconstruct.ChunkTransform()

        # store the whole metashape project (parent) meta info:
        #: the folder contains the metashape project (.psx) files
        self.project_folder = None
        #: the metashape project (file) name.
        self.project_name = None
        #: the chunk that be picked as for easyidp (this class, only deal with one chunk in metashape project)
        self.chunk_id = chunk_id
        #: the folder that contains the origial images
        self.raw_img_folder = raw_img_folder

        # hidden attributes
        self._project_chunks_dict = None
        self._chunk_id2label = {}   # for project_chunks
        self._label2chunk_id = {}   # for project_chunks
        self._reference_crs = pyproj.CRS.from_epsg(4326)

        ########################################
        # mute attributes warning for auto doc #
        ########################################
        #: project / chunk name
        self.label = self.label
        #: project meta information
        self.meta = self.meta
        #: whether this project is activated, (often for Metashape), ``<class 'bool'>``
        self.enabled = self.enabled
        #: the container for all sensors in this project (camera model), ``<class 'easyidp.Container'>``, a dict-like object consisted by :py:class:`Sensor <easyidp.reconstruct.Sensor>`
        self.sensors = self.sensors
        #: the container for all photos used in this project (images), ``<class 'easyidp.Container'>``, a dict-like object consisted by :py:class:`Photo <easyidp.reconstruct.Photo>`
        self.photos = self.photos

        self.open_project(project_path, chunk_id)

        if raw_img_folder is not None:
            self.change_photo_folder(raw_img_folder, check_img_existance=check_img_existance)

    def __repr__(self) -> str:
        return self._show_chunk()

    def __str__(self) -> str:
        return self._show_chunk()

    ###############
    # zip/xml I/O #
    ###############

    def _show_chunk(self, return_table_only=False):
        if self.project_folder is None \
            or self.project_name is None \
            or self._project_chunks_dict is None \
            or len(self._chunk_id2label) == 0 \
            or len(self._label2chunk_id) == 0:
            return "<Empty easyidp.Metashape object>"
        else:
            show_str = f"<'{self.project_name}.psx' easyidp.Metashape object with {len(self._project_chunks_dict)} active chunks>\n\n"

            head = ["id", "label"]
            data = []
            for idx, label in self._chunk_id2label.items():
                if idx == str(self.chunk_id) or label == str(self.chunk_id):
                    idx = '-> ' + idx
                data.append([idx, label])

            table_str = tabulate(data, headers=head, tablefmt='simple', colalign=["right", "left"])

            if return_table_only:
                return table_str
            else:
                return show_str + table_str
            
    def open_project(self, project_path, chunk_id=None):
        """Open a new 3D reconstructin project to overwritting current project.

        Parameters
        ----------
        project_path : str, optional
            The pix4d project file to open, like "xxxx.psx", or "xxxx" without suffix.
        chunk_id : int or str, optional
            The chunk id or name(label) want to open, by default None, open the first chunk.

        Example
        --------

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> ms = idp.Metashape()
            <Empty easyidp.Metashape object>

            >>> ms.open_project(test_data.metashape.lotus_psx)
            <'Lotus.psx' easyidp.Metashape object with 1 active chunks>

              id  label
            ----  -------
            -> 0  Chunk 1

        """
        if project_path is not None:
            self._open_whole_project(project_path)
            self.open_chunk(self.chunk_id)
        else:
            if chunk_id is not None:
                warnings.warn(
                    f"Unable to open chunk_id [{chunk_id}] for empty project with project_path={project_path}")

    def open_chunk(self, chunk_id, project_path=None):
        """switch to the other chunk, or chunk in a new project

        Parameters
        ----------
        chunk_id : int or str
            The chunk id or name(label) want to open
        project_path : str, optional
            The new metashape project file to open, like "xxxx.psx",, by default None, means swtich inside current metashape project


        Example
        --------

        Data prepare

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> ms = idp.Metashape(test_data.metashape.multichunk_psx)
            <'multichunk.psx' easyidp.Metashape object with 4 active chunks>

              id  label
            ----  --------------
            -> 1  multiple_bbb
               2  multiple_aaa
               3  multiple_aaa_1
               4  multiple_aaa_2

        Then switch from chunk 1 to chunk 4, by id or by label:

        .. code-block:: python

            >>> ms.open_chunk('4')
            # or
            >>> ms.open_chunk('multiple_aaa_2')

            >>> ms
            <'multichunk.psx' easyidp.Metashape object with 4 active chunks>
            
              id  label
            ----  --------------
               1  multiple_bbb
               2  multiple_aaa
               3  multiple_aaa_1
            -> 4  multiple_aaa_2
        
        """
        # given new project path, switch project
        if project_path is not None:  
            self._open_whole_project(project_path)
        else:
            # not given metashape project path when init this class
            if self._project_chunks_dict is None:
                raise FileNotFoundError(
                    f"Not specify Metashape project (project_path="
                    f"'{os.path.join(self.project_folder, self.project_name)}')"
                    f"Please specify `project_path` to this function"
                )
        
        if isinstance(chunk_id, int):
            chunk_id = str(chunk_id)

        if chunk_id in self._project_chunks_dict.keys():
            chunk_content_dict = read_chunk_zip(
                self.project_folder, 
                self.project_name, 
                chunk_id=chunk_id, skip_disabled=False)
            self._chunk_dict_to_object(chunk_content_dict)
        elif chunk_id in self._label2chunk_id.keys():
            chunk_content_dict = read_chunk_zip(
                self.project_folder, 
                self.project_name, 
                chunk_id=self._label2chunk_id[chunk_id], skip_disabled=False)
            self._chunk_dict_to_object(chunk_content_dict)
        else:
            raise KeyError(
                f"Could not find chunk_id [{chunk_id}] in {self._chunk_id2label}")
        
        self.chunk_id = chunk_id

    def _open_whole_project(self, project_path):
        _check_is_software(project_path)
        folder_path, project_name, ext = _split_project_path(project_path)

        # chunk_dict.keys() = [1,2,3,4,5] int id
        project_dict = read_project_zip(folder_path, project_name)

        chunk_id2label = {}
        label2chunk_id = {}
        remove_chunk_id = []
        for chunk_id in project_dict.keys():
            chunk_dict = read_chunk_zip(folder_path, project_name, chunk_id, return_label_only=True)

            if chunk_dict and chunk_dict['enabled']:
                lb = chunk_dict['label']
                lb_len = len(lb)
                # judge if two chunks have the same label
                while lb in label2chunk_id.keys():
                    # extract '_1' from 'chunk_name_1' if have
                    suffix = lb[lb_len:][1:]
                    # do not have 
                    if suffix == '':
                        lb = lb[:lb_len] + '_1'
                    else:
                        lb = lb[:lb_len] + '_' + str(int(suffix) + 1)

                chunk_id2label[chunk_id] = lb
                label2chunk_id[lb] = chunk_id
            else:   # ignore the disabled chunk.
                remove_chunk_id.append(chunk_id)
                continue

        if len(remove_chunk_id) > 0:
            for chunk_id in remove_chunk_id:
                project_dict.pop(chunk_id)

        # open the first chunk if chunk_id not given.
        first_chunk_id = list(project_dict.keys())[0]
        if len(project_dict) == 1:  # only has one chunk, open directly
            if str(self.chunk_id) not in chunk_id2label.keys() and self.chunk_id not in chunk_id2label.values():
                warnings.warn(
                    f"This project only has one chunk named "
                    f"[{first_chunk_id}] '{chunk_id2label[first_chunk_id]}', "
                    f"ignore the wrong chunk_id [{self.chunk_id}] specified by user.")
            self.chunk_id = first_chunk_id
        elif len(project_dict) > 1:   # has multiple chunks
            if self.chunk_id is None:
                warnings.warn(
                    f"The project has [{len(project_dict)}] chunks, however no chunk_id has been specified, "
                    f"open the first chunk [{first_chunk_id}] '{chunk_id2label[first_chunk_id]}' by default.")
                self.chunk_id = first_chunk_id
        else: # has zero chunk available
            raise IndexError(f"Metashape project has no chunk folder (e.g. './0/', './1/') at [{folder_path}/{project_name}.files]")

        # save to project parameters
        self.project_folder = folder_path
        self.project_name = project_name
        self._project_chunks_dict = project_dict
        self._chunk_id2label = chunk_id2label
        self._label2chunk_id = label2chunk_id


    def _chunk_dict_to_object(self, chunk_dict):
        self.label = chunk_dict["label"]
        self.enabled = chunk_dict["enabled"]

        self._reference_crs = chunk_dict["crs"]

        # adapt for empty chunk without any information
        missing_pool = []

        if "transform" in chunk_dict.keys():
            self.transform = chunk_dict["transform"]
        else:
            self.enabled = False
            missing_pool.append('transform')

        self.sensors = chunk_dict["sensors"]
        if len(chunk_dict["sensors"]) == 0:
            self.enabled = False
            missing_pool.append('sensors')

        self.photos = chunk_dict["photos"]
        if len(chunk_dict["photos"]) == 0:
            self.enabled = False
            missing_pool.append('photos')

        # show warning for emtpy tasks
        if not self.enabled:
            warnings.warn(f"Current chunk missing required {missing_pool} information "
                "(is it an empty chunk without finishing SfM tasks?) and unable to do further analysis.")
            
    def show_photo_folder(self):
        """A function to check the original photo path
        """
        print_cache = {}
        for photo in self.photos:
            root_path, photo_name = os.path.split(photo.path)
            if root_path not in print_cache.keys():
                print_cache[root_path] = []
            print_cache[root_path].append(photo_name)

        # then make the print string
        print_str = ''
        for key, value in print_cache.items():
            print_str += f"'{key}': "
            if len(value) <= 5:
                print_str += f"{value} ({len(value)} photos)"
            else:
                print_str += f"[{value[0]}, {value[1]}, ..., {value[-2]}, {value[-1]}] ({len(value)} photos)\n"

        print(print_str)
            
    def change_photo_folder(self, raw_img_folder, check_img_existance=True):
        """Change the folder path of raw images

        Parameters
        ----------
        raw_img_folder : str | dict
            The new folder path contains raw image folder.

            If type == `str` : 
                replace the root string directly.
            if type == `dict` : (not implemented)
                e.g. {'path/to/flight1/': 'new/path/to/flight1', }
        check_img_existance : bool
            Ignore the missing photos when set to False, suitable for testing project with just a few images, to avoid the FileNotFoundError
            
        """
        if isinstance(raw_img_folder, Path):
            raw_img_folder = str(raw_img_folder)
        if isinstance(raw_img_folder, str):
            if not os.path.exists(raw_img_folder) or not os.path.isdir(raw_img_folder):
                raise NotADirectoryError(f"The given folder [{raw_img_folder}] not exists")
            for photo in self.photos:
                root_path, photo_name = os.path.split(photo.path)
                new_img_path = os.path.join(raw_img_folder, photo_name)
                if check_img_existance and not os.path.exists(new_img_path):
                    raise FileNotFoundError(f"Could not find image file [{photo_name}] under given [{raw_img_folder}] folder")
                else:
                    photo.path = new_img_path
        elif isinstance(raw_img_folder, dict):
            raise NotImplementedError("Changing raw_img_folder by dictionary has not supported yet.")
        else:
            raise TypeError(f"Only <str> or <dict> type are acceptable, not <{type(raw_img_folder)}> for {raw_img_folder}")

    
    #######################
    # backward projection #
    #######################

    def _local2world(self, points_np):
        return apply_transform_matrix(points_np, self.transform.matrix)

    def _world2local(self, points_np):
        if self.transform.matrix_inv is None:
            self.transform.matrix_inv = np.linalg.inv(self.transform.matrix)

        return apply_transform_matrix(points_np, self.transform.matrix_inv)

    def _world2crs(self, points_np):
        if self.crs is None:
            return idp.geotools.convert_proj3d(points_np, self._world_crs, self._reference_crs)
        else:
            return idp.geotools.convert_proj3d(points_np, self._world_crs, self.crs)

    def _crs2world(self, points_np):
        if self.crs is None:
            return idp.geotools.convert_proj3d(points_np, self._reference_crs, self._world_crs)
        else:
            return idp.geotools.convert_proj3d(points_np, self.crs, self._world_crs)

    def _back2raw_one2one(self, points_np, photo_id, distortion_correct=True):
        """Project one ROI(polygon) on one given photo

        Parameters
        ----------
        points_np : ndarray (nx3)
            The 3D coordinates of polygon vertexm, in local coordinates
        photo_id : int | str | <easyidp.Photo> object
            the photo that will project on. Can be photo id (int), photo name (str) or <Photo> object
        distortion_correct : bool, optional
            Whether do distortion correction, by default True (back to raw image);
            If back to software corrected images without len distortion, set it to True. 
            Pix4D support do this operation, seems metashape not supported yet.

        Returns
        -------
        ndarray
            nx2 pixel coordiante of ROI on images
        """
        if isinstance(photo_id, (int, str)):
            camera_i = self.photos[photo_id]
            sensor_i = self.sensors[camera_i.sensor_id]
        elif isinstance(photo_id, idp.reconstruct.Photo):
            camera_i = photo_id
            sensor_i = photo_id.sensor
        else:
            raise TypeError(
                f"Only <int> photo id or <easyidp.reconstruct.Photo> object are accepted, "
                f"not {type(photo_id)}")

        if not camera_i.enabled:
            return None
        
        t = camera_i.transform[0:3, 3]
        r = camera_i.transform[0:3, 0:3]

        points_np, is_single = idp.geotools.is_single_point(points_np)

        xyz = (points_np - t).dot(r)

        xh = xyz[:, 0] / xyz[:, 2]
        yh = xyz[:, 1] / xyz[:, 2]

        # without distortion
        if distortion_correct:
                        # with distortion
            u, v = sensor_i.calibration.calibrate(xh, yh)

            out = np.vstack([u, v]).T

        else:
            w = sensor_i.width
            h = sensor_i.height
            f = sensor_i.calibration.f
            cx = sensor_i.calibration.cx
            cy = sensor_i.calibration.cy

            k = np.asarray([[f, 0, w / 2 + cx, 0],
                            [0, f, h / 2 + cy, 0],
                            [0, 0, 1, 0]])

            # make [x, y, 1, 1] for multiple points
            pch = np.vstack([xh, yh, np.ones(len(xh)), np.ones(len(xh))]).T
            ppix = pch.dot(k.T)

            out = ppix[:, 0:2]

        if is_single:
            return out[0, :]
        else:
            return out

    def back2raw_crs(self, points_xyz, ignore=None, log=False):
        """Projs one GIS coordintates ROI (polygon) to all images

        Parameters
        ----------
        points_hv : ndarray (nx3)
            The 3D coordinates of polygon vertexm, in CRS coordinates
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
            a dictionary that key = img_name and values= pixel coordinate

        Example
        -------

        Data preparation

        .. code-block:: python

            >>> import easyidp as idp

            >>> ms = idp.Metashape(test_data.metashape.lotus_psx)
            >>> dsm = idp.GeoTiff(test_data.metashape.lotus_dsm)
            >>> ms.crs = dsm.crs

            >>> plot =  np.array([   # N1E1 plot geo coordinate
            ...     [ 368020.2974959 , 3955511.61264302,      97.56272272],
            ...     [ 368022.24288365, 3955512.02973983,      97.56272272],
            ...     [ 368022.65361232, 3955510.07798313,      97.56272272],
            ...     [ 368020.69867274, 3955509.66725421,      97.56272272],
            ...     [ 368020.2974959 , 3955511.61264302,      97.56272272]
            ... ])

        ..caution:: specifying the CRS of metashape project (``ms.crs = dsm.crs``) is required before doing backward projection calculation

        Then use this function to find the previous ROI positions on the raw images:

        .. code-block:: python

            >>> out_dict = ms.back2raw_crs(plot)

            >>> out_dict["DJI_0478"]
            array([[   2.03352333, 1474.90817792],
                   [  25.04572582, 1197.08827224],
                   [ 311.99971438, 1214.81701547],
                   [ 288.88669685, 1492.59824542],
                   [   2.03352333, 1474.90817792]])

        """
        if not self.enabled:
            raise TypeError("Unable to process disabled chunk (.enabled=False)")
        
        if self.crs is None:
            warnings.warn("Have not specify the CRS of output DOM/DSM/PCD, may get wrong backward projection results, please specify it by `ms.crs=dom.crs` or `ms.crs=pyproj.CRS.from_epsg(...)` ")
        
        if self.crs is not None and self.crs.name in ['Local Coordinates', 'Local Coordinates (m)']:
            local_coord = self._world2local(points_xyz)
        else:
            local_coord = self._world2local(self._crs2world(points_xyz))

        if log:
            print(f'[Calculator][Judge]camera_name photo.width photo.height -> x.min \t x.max \t y.min \t y.max')

        # for each raw image in the project / flight
        out_dict = {}
        for photo_name, photo in self.photos.items():
            # skip not enabled photos
            if not photo.enabled:
                continue
            # reverse projection to given raw images
            projected_coord = self._back2raw_one2one(local_coord, photo, distortion_correct=True)

            # find out those correct images
            if log:
                print(f'[Calculator][Judge]{photo.label}w:{photo.sensor.width}h:{photo.sensor.height}->', end='')

            coords = photo.sensor.in_img_boundary(projected_coord, ignore=ignore, log=log)
            if coords is not None:
                out_dict[photo_name] = coords

        return out_dict


    def back2raw(self, roi, save_folder=None, **kwargs):
        """Projects several GIS coordintates ROIs (polygons) to all images

        Parameters
        ----------
        roi : easyidp.ROI | dict
            the <ROI> object created by easyidp.ROI() or dictionary
        save_folder : str, optional
            the folder to save json files and parts of ROI on raw images, by default None
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

            >>> ms = idp.Metashape(project_path=lotus.metashape.project)

            >>> roi = idp.ROI(lotus.shp, name_field=0)
            [shp][proj] Use projection [WGS 84] for loaded shapefile [plots.shp]
            Read shapefile [plots.shp]: 100%|███████████████| 112/112 [00:00<00:00, 2481.77it/s]
            >>> roi = roi[0:2]

            >>> roi.get_z_from_dsm(lotus.pix4d.dsm)


        Then using this function to do backward projection:

        .. code-block:: python

            >>> out_all = ms.back2raw(roi)
            {
                'N1W1': {
                    'DJI_0478.JPG': 
                        array([[  14.96726711, 1843.13937997],
                               [  38.0361733 , 1568.36113526],
                               [ 320.25420037, 1584.28772847],
                               [ 297.16110119, 1859.05913936],
                               [  14.96726711, 1843.13937997]])
                    'DJI_0479':
                        array([...])
                    ...
                }
                'N1W2': {...}   # output of `back2raw_crs()`
            }

        """
        if not self.enabled:
            raise TypeError("Unable to process disabled chunk (.enabled=False)")
        
        if self.crs is None and roi.crs is None:
            warnings.warn("Have not specify the CRS of output DOM/DSM/PCD, may get wrong backward projection results, please specify it by either `ms.crs=...` or `roi.crs=...` ")
        
        out_dict = {}

        before_crs = ccopy(self.crs)
        self.crs = ccopy(roi.crs)

        pbar = tqdm(roi.items(), desc=f"Backward roi to raw images")
        for k, points_xyz in pbar:
            if isinstance(save_folder, str) and os.path.isdir(save_folder):
                save_path = os.path.join(save_folder, k)
            else:
                save_path = None

            if points_xyz.shape[1] != 3:
                raise ValueError(f"The back2raw function requires 3D roi with shape=(n, 3), but [{k}] is {points_xyz.shape}")

            one_roi_dict= self.back2raw_crs(points_xyz, **kwargs)

            out_dict[k] = one_roi_dict

        self.crs = before_crs

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
        
            >>> import numpy as np
            >>> np.set_printoptions(suppress=True)

            >>> import easyidp as idp

            >>> lotus = idp.data.Lotus()
            >>> ms = idp.Metashape(project_path=lotus.metashape.project)

        Then use this function to get the photo position in 3D world:

        .. code-block:: python

            >>> out = ms.get_photo_position()
            {
                'DJI_0422': array([139.54053245,  35.73458169, 130.09433649]), 
                'DJI_0423': array([139.54053337,  35.73458315, 129.93437641]),
                ...
            }

        .. caution:: by default, if not specifying the CRS of metashape project, it will return in default CRS (epsg: 4326) -> (lon, lat, height), if need turn to the same coordinate like DOM/DSM, please specify the CRS first

        .. code-block:: python

            >>> dom = idp.GeoTiff(lotus.metashape.dom)
            >>> ms.crs = dom.crs

            >>> out = ms.get_photo_position()
            {
                'DJI_0422': array([ 368017.73174354, 3955492.1925972 ,     130.09433649]), 
                'DJI_0423': array([ 368017.81717717, 3955492.35300323,     129.93437641]),
                ...
            }

        """
        if not self.enabled:
            raise TypeError("Unable to process disabled chunk (.enabled=False)")
    
        if self._photo_position_cache is not None and not refresh:
            return self._photo_position_cache.copy()
        else:
            # change the out crs
            before_crs = ccopy(self.crs)
            if isinstance(to_crs, pyproj.CRS) and not to_crs.equals(self.crs):
                self.crs = ccopy(to_crs)

            out = {}
            pbar = tqdm(self.photos, desc=f"Getting photo positions")
            for p in pbar:
                if p.enabled:
                    # the metashape logic, did the convertion based on self.crs in the following functions
                    # this is different with the pix4d project, check Pix4D.get_photo_position() for more info
                    pos = self._world2crs(self._local2world(p.transform[0:3, 3]))

                    out[p.label] = pos
                    p.position = pos

            self.crs = before_crs

            # cache the photo position results for later use
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

            >>> out_all = ms.back2raw(roi)
            {
                'N1W1': {
                    'DJI_0478.JPG': 
                        array([[  14.96726711, 1843.13937997],
                               [  38.0361733 , 1568.36113526],
                               [ 320.25420037, 1584.28772847],
                               [ 297.16110119, 1859.05913936],
                               [  14.96726711, 1843.13937997]])
                    'DJI_0479':
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

            >>> img_dict_sort = ms.sort_img_by_distance(
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

            >>> img_dict_sort = ms.sort_img_by_distance(
            ...     out_all, roi,
            ...     num=1   # only keep the closest images
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

        """
        if not self.enabled:
            raise TypeError("Unable to process disabled chunk (.enabled=False)")
        
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
            the image file name, by default None, plotting all available images
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
            raise FileNotFoundError(f"Could not find the image file [{img_name}] in the Metashape project")
        
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

###############
# zip/xml I/O #
###############

def read_project_zip(project_folder, project_name):
    """parse xml in the ``project.zip`` file, and get the chunk id and path

    Parameters
    ----------
    project_folder: str
    project_name: str

    Returns
    -------
    project_dict: dict
        key = chunk_id, value = chunk_path

    Notes
    -----
    If one project path look likes: ``/root/to/metashape/test_proj.psx``, 
    then the input parameter should be:
    
    - ``project_folder = "/root/to/metashape/"``
    - ``project_name = "test_proj"``

    And obtained xml_str example:

    .. code-block:: xml

        <document version="1.2.0">
          <chunks next_id="2">
            <chunk id="0" path="0/chunk.zip"/>
          </chunks>
          <meta>
            <property name="Info/LastSavedDateTime" value="2020:06:22 02:23:20"/>
            <property name="Info/LastSavedSoftwareVersion" value="1.6.2.10247"/>
            <property name="Info/OriginalDateTime" value="2020:06:22 02:20:16"/>
            <property name="Info/OriginalSoftwareName" value="Agisoft Metashape"/>
            <property name="Info/OriginalSoftwareVendor" value="Agisoft"/>
            <property name="Info/OriginalSoftwareVersion" value="1.6.2.10247"/>
          </meta>
        </document>

    Example
    --------

    Data prepare

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> project_folder = test_data.metashape.lotus_psx.parents[0]
        PosixPath('/Users/<user>/Library/Application Support/easyidp.data/data_for_tests/metashape')
        >>> project_name = 'Lotus'

    Then use this function to 

    .. code-block:: python

        >>> idp.metashape.read_project_zip(project_folder, project_name)
        {'0': '0/chunk.zip'}

    """
    project_dict = {}

    zip_file = f"{project_folder}/{project_name}.files/project.zip"
    xml_str = _get_xml_str_from_zip_file(zip_file, "doc.xml")
    xml_tree = ElementTree.fromstring(xml_str)

    for chunk in xml_tree[0]:   # tree[0] -> <chunks>
        project_dict[chunk.attrib['id']] = chunk.attrib['path']

    return project_dict


def read_chunk_zip(project_folder, project_name, chunk_id, skip_disabled=False, return_label_only=False, raw_img_folder=None):
    """parse xml in the given ``chunk.zip`` file.

    Parameters
    ----------
    project_folder: str
    project_name: str
    chunk_id: int or str
        the chunk id start from 0 of chunk.zip
    skip_disabled: bool
        return None if chunk enabled is False in metashape project
    return_label_only: bool
        Only parse chunk.label, by default False

    Returns
    -------
    chunk_dict or None

    Notes
    -----
    If one project path look likes: ``/root/to/metashape/test_proj.psx``, 
    then the input parameter should be:
    
    - ``project_folder = "/root/to/metashape/"``
    - ``project_name = "test_proj"``

    Example for xml_str:

    .. code-block:: xml

        <chunk version="1.2.0" label="170525" enabled="true">
        <sensors next_id="1">
            <sensor id="0" label="FC550, DJI MFT 15mm F1.7 ASPH (15mm)" type="frame">
            ...  -> _decode_sensor_tag()
            </sensor>
            ...  -> for loop in this function
        </sensors>

        <cameras next_id="266" next_group_id="0">
            <camera id="0" sensor_id="0" label="DJI_0151">
            ... -> _decode_camera_tag()
            </camera>
            ...  -> for loop in this function
        </cameras>

        <frames next_id="1">
            <frame id="0" path="0/frame.zip"/>
        </frames>

        <reference>GEOGCS["WGS 84",DATUM["World Geodetic System 1984",], ..., AUTHORITY["EPSG","4326"]]</reference>

        <transform>
            ...  -> _decode_chunk_transform_tag()
        <transform>

        <region>
            <center>-8.1800672545192263e+00 -2.6103071594817338e+00 -1.0706980639382815e+01</center>
            <size>3.3381093645095824e+01 2.3577857828140260e+01 7.2410767078399658e+00</size>
            <R>-9.8317886026354417e-01 ...  9.9841729020145376e-01</R>   // 9 numbers
        </region>

    Example
    --------

    Data prepare

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> project_folder = test_data.metashape.lotus_psx.parents[0]
        PosixPath('/Users/<user>/Library/Application Support/easyidp.data/data_for_tests/metashape')
        >>> project_name = 'Lotus'

    Then parse the chunk:

    .. code-block:: python

        >>> idp.metashape.read_chunk_zip(project_folder, project_name, chunk_id=0)
        {
            'label': 'Chunk 1', 

            'enabled': True, 

            'transform': <easyidp.reconstruct.ChunkTransform object at 0x7fe2b81bf370>,

            'sensors': <easyidp.Container> with 1 items
                       [0]     FC550, DJI MFT 15mm F1.7 ASPH (15mm)
                       <easyidp.reconstruct.Sensor object at 0x7fe2a873a040>, 

            'photos': <easyidp.Container> with 151 items
                      [0]     DJI_0422
                      <easyidp.reconstruct.Photo object at 0x7fe2a873a9a0>
                      [1]     DJI_0423
                      <easyidp.reconstruct.Photo object at 0x7fe2a873a130>
                      ...
                      [149]   DJI_0571
                      <easyidp.reconstruct.Photo object at 0x7fe298e82820>
                      [150]   DJI_0572
                      <easyidp.reconstruct.Photo object at 0x7fe298e82850>, 
            
            'crs': <Geographic 2D CRS: EPSG:4326>
                   Name: WGS 84
                   Axis Info [ellipsoidal]:
                   - Lat[north]: Geodetic latitude (degree)
                   - Lon[east]: Geodetic longitude (degree)
                   Area of Use:
                   - name: World.
                   - bounds: (-180.0, -90.0, 180.0, 90.0)
                   Datum: World Geodetic System 1984 ensemble
                   - Ellipsoid: WGS 84
                   - Prime Meridian: Greenwich
        }

    """
    frame_zip_file = f"{project_folder}/{project_name}.files/{chunk_id}/chunk.zip"
    # for test data, some metashape projects are not complete, then skip and return None
    if not os.path.exists(frame_zip_file):
        print(f'[Warning] Metashape project {project_folder} Chunk {chunk_id} folder missing')
        return None

    xml_str = _get_xml_str_from_zip_file(frame_zip_file, "doc.xml")
    xml_tree = ElementTree.fromstring(xml_str)

    chunk_dict = {}

    chunk_dict["label"] = xml_tree.attrib["label"]
    chunk_dict["enabled"] = bool(xml_tree.attrib["enabled"])

    if skip_disabled and not chunk_dict["enabled"]:
        return None

    if return_label_only:
        return chunk_dict

    # metashape chunk.transform.matrix
    transform_tags = xml_tree.findall("./transform")
    if len(transform_tags) == 1:
        chunk_dict["transform"] = _decode_chunk_transform_tag(transform_tags[0])

    # change sensor xml to idp object
    debug_meta = {
        "project_folder": project_folder, 
        "project_name"  : project_name,
        "chunk_id": chunk_id,
        "chunk_path": frame_zip_file
    }
    sensors = _sensorxml2object(
        xml_tree, debug_meta)
    chunk_dict["sensors"] = sensors

    # change photo xml to idp object
    photos = _photoxml2object(
        xml_tree, sensors
    )
    chunk_dict["photos"] = photos

    for frame_tag in xml_tree.findall("./frames/frame"):
        # frame_zip_idx = frame_tag.attrib["id"]
        frame_zip_path = frame_tag.attrib["path"]

        frame_zip_file = f"{project_folder}/{project_name}.files/{chunk_id}/{frame_zip_path}"
        frame_xml_str = _get_xml_str_from_zip_file(frame_zip_file, "doc.xml")

        camera_meta, marker_meta = _decode_frame_xml(frame_xml_str)
        for camera_idx, camera_path in camera_meta.items():
            chunk_dict["photos"][camera_idx]._path = camera_path
            # here need to resolve absolute path
            # <photo path="../../../../source/220613_G_M600pro/DSC06035.JPG">
            # this is the root to 220613_G_M600pro.files\0\0\frame.zip"
            if "../../../" in camera_path:
                chunk_dict["photos"][camera_idx].path = idp.parse_relative_path(
                    frame_zip_file, camera_path
                )
            else:
                chunk_dict["photos"][camera_idx].path = camera_path

    chunk_dict["crs"] = _decode_chunk_reference_tag(xml_tree.findall("./reference"))

    return chunk_dict

def _sensorxml2object(xml_tree, debug_meta):
    sensors = idp.Container()
    sensor_total_num = int(xml_tree.findall("./sensors")[0].attrib['next_id'])
    for i in range(sensor_total_num):
        empty_sensor = idp.reconstruct.Sensor()
        empty_sensor.id = i
        empty_sensor.label = f"empty {i}"
        sensors[i] = empty_sensor

    for sensor_tag in xml_tree.findall("./sensors/sensor"):
        sensor = _decode_sensor_tag(sensor_tag, debug_meta)
        if sensor.calibration is not None:
            sensor.calibration.software = "metashape"

        # add unique key for duplicate labels
        if sensor.label in sensors.item_label.keys():
            sensor.label = f"{sensor.label} [{sensor.id}]"

        sensors[sensor.id] = sensor

    return sensors

def _photoxml2object(xml_tree, sensors):
    photos = idp.Container()
    # when no camera groups, the tag is 
    # <cameras ... >
    #   <camera ... >
    #
    # But if has camera groups, the tag is 
    # <cameras next_id="218" next_group_id="3">
    #   <group id="0" label="100MEDIA" type="folder">
    #     <camera id="0" sensor_id="0" component_id="0" label="DJI_0003">
    #
    # for multispectral projects, both camera and group existing
    # need to do at the same time
    # <cameras next_id="218" next_group_id="3">
    #   <group id="0" label="100MEDIA" type="folder">
    #     <camera id="0" sensor_id="0" component_id="0" label="DJI_0003">
    #     <camera id="0" sensor_id="0" component_id="0" master_id="932" label="DJI_0003">
    #     <camera id="0" sensor_id="0" component_id="0" master_id="932" label="DJI_0003">
    #     <camera id="0" sensor_id="0" component_id="0" master_id="932" label="DJI_0003">
    #   </group>
    #   <camera ... >
    #   <camera master_id="932" ... >
    #   <camera master_id="932" ... >
    #   <camera master_id="932" ... >
    #   <camera ... >
    group_tags  = xml_tree.findall("./cameras/group")
    camera_tags = xml_tree.findall("./cameras/camera")

    # create an empty conatiner for disordered camera
    cam_total_num = int(xml_tree.findall("./cameras")[0].attrib['next_id'])
    for i in range(cam_total_num):
        disabled_camera = idp.reconstruct.Photo()
        disabled_camera.id = i
        disabled_camera.label = f'empty {i}'
        disabled_camera.enabled = False
        photos[i] = disabled_camera

    ######################
    # decode camera tags #
    ######################
    for camera_tag in camera_tags:
        camera = _decode_camera_tag(camera_tag)
        camera.sensor = sensors[camera.sensor_id]
        # multispectral camera groups, get transform from master_id
        if camera.master_id is not None:
            camera.transform = photos[camera.master_id].transform
        photos[camera.id] = camera

    #########################
    # decode container tags #
    #########################
    # judge if has group with the same name
    group_label_pool = [g.attrib['label'] for g in group_tags]
    group_label_pool_unique = set(group_label_pool)
    if len(group_label_pool_unique) != len(group_label_pool):
        has_duplicate_name = True
    else:
        has_duplicate_name = False

    for group_tag in group_tags:
        if has_duplicate_name:
            group_label = f"[{group_tag.attrib['id']}]{group_tag.attrib['label']}"
        else:
            group_label = group_tag.attrib['label'] 
        camera_tags = group_tag.findall("./camera")

        for camera_tag in camera_tags:
            camera = _decode_camera_tag(camera_tag)
            camera.sensor = sensors[camera.sensor_id]
            # rename camera label to group-label to avoid duplicates
            camera.label = f"{group_label}-{camera.label}" 
            photos[camera.id] = camera

    return photos

def _split_project_path(path: str):
    """Get project name, current folder, extension, etc. from given project path.

    Parameters
    ----------
    path: str
        e.g. proj_path="/root/to/metashape/test_proj.psx"

    Returns
    -------
    folder_path: str
        e.g. "/root/to/metashape/"
    project_name: str
        e.g. "test_proj"
    ext: str:
        e.g. "psx"
    """
    folder_path, file_name = os.path.split(path)
    project_name, ext = os.path.splitext(file_name)

    return folder_path, project_name, ext


def _check_is_software(path: str):
    """Check if given path is metashape project structure

    Parameters
    ----------
    path: str
        e.g. proj_path="/root/to/metashape/test_proj.psx"

    Returns
    -------
    raise error if not a metashape projects (missing some files or path not found)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find Metashape project file [{path}]")

    folder_path, project_name, ext = _split_project_path(path)
    data_folder = os.path.join(folder_path, project_name + ".files")

    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Could not find Metashape project file [{data_folder}]")


def _get_xml_str_from_zip_file(zip_file, xml_file):
    """read xml file in zip file

    Parameters
    ----------
    zip_file: str
        the path to zip file, e.g. "data/metashape/goya_test.files/project.zip"
    xml_file: str
        the file name in zip file, e.g. "doc.xml" in previous zip file

    Returns
    -------
    xml_str: str
        the string of readed xml file
    """
    with zipfile.ZipFile(zip_file) as zfile:
        xml = zfile.open(xml_file)
        xml_str = xml.read().decode("utf-8")

    return xml_str


def _decode_chunk_transform_tag(xml_obj):
    """
    Topic: Camera coordinates to world coordinates using 4x4 matrix
    https://www.agisoft.com/forum/index.php?topic=6176.15

    Parameters
    ----------
    xml_obj: xml.etree.ElementTree() object
        one element of xml_tree.findall("./transform")

        <transform>
          <rotation locked="false">-9.9452052163852955e-01 ...  -5.0513659345945017e-01</rotation>  // 9 numbers
          <translation locked="false">7.6503412293334154e+00 ... -1.8356149553667311e-01</translation>   // 3 numbers
          <scale locked="true">8.7050086657310788e-01</scale>
        </transform>

    Returns
    -------
    transform_dict: dict
       key: ["rotation", "translation", "scale", "transform"]
    """
    transform = idp.reconstruct.ChunkTransform()
    chunk_rotation_str = xml_obj.findall("./rotation")[0].text
    transform.rotation = np.fromstring(chunk_rotation_str, sep=" ", dtype=float).reshape((3, 3))

    chunk_translation_str = xml_obj.findall("./translation")[0].text
    transform.translation = np.fromstring(chunk_translation_str, sep=" ", dtype=float)

    transform.scale = float(xml_obj.findall("./scale")[0].text)

    transform.matrix = np.zeros((4, 4))
    transform.matrix[0:3, 0:3] = transform.rotation * transform.scale
    transform.matrix[0:3, 3] = transform.translation
    transform.matrix[3, :] = np.asarray([0, 0, 0, 1])

    return transform


def _decode_chunk_reference_tag(xml_obj):
    """
    change CRS string to pyproj.CRS object
    Parameters
    ----------
    xml_obj: xml.etree.ElementTree() object
        if no GPS info provided, the reference tag will look like this:
        <reference>LOCAL_CS["Local Coordinates (m)",LOCAL_DATUM["Local Datum",0],
        UNIT["metre",1,AUTHORITY["EPSG","9001"]]]</reference>

        if given longitude and latitude, often is "WGS84":
          <reference>GEOGCS["WGS 84",DATUM["World Geodetic System 1984",SPHEROID["WGS 84",6378137,298.
          257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM
          ["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG",
          "9102"]],AUTHORITY["EPSG","4326"]]</reference>

    Returns
    -------

    """
    crs_str = xml_obj[0].text
    # metashape default value
    local_crs = 'LOCAL_CS["Local Coordinates (m)",LOCAL_DATUM["Local Datum",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]'

    if crs_str == local_crs:
        crs_obj = pyproj.CRS.from_dict({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'})
    else:
        crs_obj = pyproj.CRS.from_string(crs_str)

    '''
    sometimes, the Photoscan given CRS string can not transform correctly
    this is the solution for "WGS 84" CRS shown in the previous example
    
    <reference>GEOGCS["WGS 84",DATUM["World Geodetic System 1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9102"]],AUTHORITY["EPSG","4326"]]</reference>
    
    only works when pyproj==2.6.1.post
    
    when pyproj==3.3.1
    crs_obj.datum = 
      DATUM["World Geodetic System 1984",
      ELLIPSOID["WGS 84",6378137,298.257223563,
            LENGTHUNIT["metre",1]],
      ID["EPSG",6326]]
    pyproj.CRS.from_epsg(4326).datum = 
      ENSEMBLE["World Geodetic System 1984 ensemble",
      MEMBER["World Geodetic System 1984 (Transit)",
          ID["EPSG",1166]],
      MEMBER["World Geodetic System 1984 (G730)",
          ID["EPSG",1152]],
      MEMBER["World Geodetic System 1984 (G873)",
          ID["EPSG",1153]],
      MEMBER["World Geodetic System 1984 (G1150)",
          ID["EPSG",1154]],
      MEMBER["World Geodetic System 1984 (G1674)",
          ID["EPSG",1155]],
      MEMBER["World Geodetic System 1984 (G1762)",
          ID["EPSG",1156]],
      MEMBER["World Geodetic System 1984 (G2139)",
          ID["EPSG",1309]],
      ELLIPSOID["WGS 84",6378137,298.257223563,
          LENGTHUNIT["metre",1],
          ID["EPSG",7030]],
      ENSEMBLEACCURACY[2.0],
      ID["EPSG",6326]]
    '''
    crs_wgs84 = pyproj.CRS.from_epsg(4326)

    obj_d = crs_obj.datum.to_json_dict()
    # {'$schema': 'https://proj.org/sch...chema.json', 
    # 'type': 'GeodeticReferenceFrame', 
    # 'name': 'World Geodetic System 1984', 
    # 'ellipsoid': {'name': 'WGS 84', 'semi_major_axis': 6378137, 'inverse_flattening': 298.257223563}, 
    # 'id': {'authority': 'EPSG', 'code': 6326}}

    wgs84_d = crs_wgs84.datum.to_json_dict()
    # {'$schema': 'https://proj.org/sch...chema.json', 
    # 'type': 'DatumEnsemble', 
    # 'name': 'World Geodetic Syste...4 ensemble', 
    # 'members': [{...}, {...}, {...}, {...}, {...}, {...}, {...}], 
    # 'ellipsoid': {'name': 'WGS 84', 'semi_major_axis': 6378137, 'inverse_flattening': 298.257223563}, 
    # 'accuracy': '2.0', 
    # 'id': {'authority': 'EPSG', 'code': 6326}}

    if  obj_d["id"] == wgs84_d["id"] and \
        obj_d["ellipsoid"] == wgs84_d["ellipsoid"]:
        return crs_wgs84
    else:
        return crs_obj


def _decode_sensor_tag(xml_obj, debug_meta={}):
    """
    Parameters
    ----------
    xml_obj: xml.etree.ElementTree() object
        one element of xml_tree.findall("./sensors/sensor")

        <sensor id="0" label="FC550, DJI MFT 15mm F1.7 ASPH (15mm)" type="frame">
          <resolution width="4608" height="3456"/>
          <property name="pixel_width" value="0.00375578257860832"/>
          <property name="pixel_height" value="0.00375578257860832"/>
          <property name="focal_length" value="15"/>
          <property name="layer_index" value="0"/>
          <bands>
            <band label="Red"/>
            <band label="Green"/>
            <band label="Blue"/>
          </bands>
          <data_type>uint8</data_type>
          <calibration type="frame" class="adjusted">
            ...  -> _decode_calibration_tag()
          </calibration>
          <covariance>
            <params>f cx cy k1 k2 k3 p1 p2</params>
            <coeffs>...</coeffs>
          </covariance>
        </sensor>

    Returns
    -------
    sensor: easyidp.Sensor object
    """
    sensor = idp.reconstruct.Sensor()

    sensor.id = int(xml_obj.attrib["id"])
    sensor.label = xml_obj.attrib["label"]
    sensor.type = xml_obj.attrib["type"]

    resolution = xml_obj.findall("./resolution")[0]
    sensor.width = int(resolution.attrib["width"])
    sensor.width_unit = "px"
    sensor.height = int(resolution.attrib["height"])
    sensor.height_unit = "px"

    sensor.pixel_width = float(xml_obj.findall("./property/[@name='pixel_width']")[0].attrib["value"])
    sensor.pixel_width_unit = "mm"
    sensor.pixel_height = float(xml_obj.findall("./property/[@name='pixel_height']")[0].attrib["value"])
    sensor.pixel_height_unit = "mm"
    sensor.focal_length = float(xml_obj.findall("./property/[@name='focal_length']")[0].attrib["value"])

    calib_tags = xml_obj.findall("./calibration")

    # check if has <calibration type="frame" class="adjusted"> tag
    has_adjusted_tag = False
    for c in calib_tags:
        if c.attrib['class'] == 'adjusted':
            has_adjusted_tag = True
            sensor.calibration = _decode_calibration_tag(c)
            sensor.calibration.sensor = sensor
            break

    if len(calib_tags) != 1:
        if has_adjusted_tag:
            warnings.warn(f'Detect {len(calib_tags)} <calibration> tags in <sensor label={sensor.label}> tag, using <calibration class="adjusted">')
            

    if not has_adjusted_tag:
        xml_str = minidom.parseString(ElementTree.tostring(xml_obj)).toprettyxml(indent="  ")
        # remove the first line <?xml version="1.0" ?> and empty lines
        xml_str = os.linesep.join([s for s in xml_str.splitlines() if s.strip() and '?xml version=' not in s])
        warnings.warn(
            f'No expected <calibration class="adjusted"> tag found in <sensor label={sensor.label}> tag\n'
            f'Problemed XML tags for debugging reference: \n{xml_str}\n')
        
        sensor.calibration = None

    return sensor


def _decode_calibration_tag(xml_obj):
    """
    Parameters
    ----------
    xml_obj: xml.etree.ElementTree() object
        one element of sensor_tag.findall("./calibration")
    Returns
    -------
    calibration: easyidp.Calibration object

    Notes
    -----

    Calibration tag example 1:

    .. code-block:: xml

        <calibration type="frame" class="adjusted">
            <resolution width="5472" height="3648"/>
            <f>3648</f>
            <k1>-0.01297256557769369</k1>
            <k2>0.00071786332278828374</k2>
            <k3>0.007914514308754287</k3>
            <p1>-0.0011379213280524752</p1>
            <p2>-0.0014457166089453365</p2>
        </calibration>

    Calibration tag example 2:

    .. code-block:: xml

        <calibration type="frame" class="adjusted">
            <resolution width="4608" height="3456"/>
            <f>4317.82144109411</f>
            <cx>54.3448479006764</cx>
            <cy>3.5478801249494</cy>
            <k1>0.0167015604921295</k1>
            <k2>-0.0239416622459823</k2>
            <k3>0.0363031944008484</k3>
            <p1>0.0035044847840485</p1>
            <p2>0.00103698463015268</p2>
        </calibration>

    """
    calibration = idp.reconstruct.Calibration()

    calibration.f = float(xml_obj.findall("./f")[0].text)
    calibration.f_unit = "px"

    try:
        calibration.cx = float(xml_obj.findall("./cx")[0].text)
    except IndexError:  # example 1, no cx tag
        calibration.cx = 0
    calibration.cx_unit = "px"

    try:
        calibration.cy = float(xml_obj.findall("./cy")[0].text)
    except IndexError:  # example 1, no cx tag
        calibration.cy = 0
    calibration.cy_unit = "px"

    if len(xml_obj.findall("./b1")) == 1:
        calibration.b1 = float(xml_obj.findall("./b1")[0].text)
    if len(xml_obj.findall("./b2")) == 1:
        calibration.b2 = float(xml_obj.findall("./b2")[0].text)

    if len(xml_obj.findall("./k1")) == 1:
        calibration.k1 = float(xml_obj.findall("./k1")[0].text)
    if len(xml_obj.findall("./k2")) == 1:
        calibration.k2 = float(xml_obj.findall("./k2")[0].text)
    if len(xml_obj.findall("./k3")) == 1:
        calibration.k3 = float(xml_obj.findall("./k3")[0].text)
    if len(xml_obj.findall("./k4")) == 1:
        calibration.k4 = float(xml_obj.findall("./k4")[0].text)

    if len(xml_obj.findall("./p1")) == 1:
        calibration.t1 = float(xml_obj.findall("./p1")[0].text)
    if len(xml_obj.findall("./p2")) == 1:
        calibration.t2 = float(xml_obj.findall("./p2")[0].text)
    if len(xml_obj.findall("./p3")) == 1:
        calibration.t3 = float(xml_obj.findall("./p3")[0].text)
    if len(xml_obj.findall("./p4")) == 1:
        calibration.t4 = float(xml_obj.findall("./p4")[0].text)

    return calibration


def _decode_camera_tag(xml_obj):
    """
    Parameters
    ----------
    xml_obj: xml.etree.ElementTree() object
        one element of xml_tree.findall("./cameras/camera")

    Notes
    -----

    Camera Tag example 1:

    .. code-block:: xml

        <camera id="0" sensor_id="0" label="DJI_0151">
            <transform>
            0.99893511, -0.04561155,  0.00694542, -5.50542042
            -0.04604262, -0.9951647 ,  0.08676   , 13.25994938
            0.00295458, -0.0869874 , -0.99620503,  2.15491524
            0.        ,  0.        ,  0.        ,  1.         
            </transform>  // 16 numbers
            <rotation_covariance>5.9742650282250832e-04 ... 2.3538470709659123e-04</rotation_covariance>  // 9 numbers
            <location_covariance>2.0254219245789448e-02 ... 2.6760756179895751e-02</location_covariance>  // 9 numbers
            <orientation>1</orientation>

            // sometimes have following reference tag, otherwise need to look into frames.zip xml
            <reference x="139.540561166667" y="35.73454525" z="134.765" yaw="164.1" pitch="0"
                        roll="-0" enabled="true" rotation_enabled="false"/>
        </camera>

    Camera tag example 2:
    
    some camera have empty tags, also need to deal with such situation

    .. code-block:: xml

        <camera id="254" sensor_id="0" label="DJI_0538.JPG">
          <orientation>1</orientation>
        </camera>

    Camera tag example 3:

    .. code-block:: xml

        <camera id="62" sensor_id="0" label="DJI_0121">
            <orientation>1</orientation>
            <reference x="139.54051557" y="35.739036839999997" z="106.28" yaw="344.19999999999999" pitch="0" roll="-0" enabled="true" rotation_enabled="false"/>
        </camera>

    Camera tag example 4:

    The multi-spectral camera bind other band to one main bind, which produced an extra :code:`master_id` tag

    .. code-block:: xml

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

    Returns
    -------
    camera: easyidp.Photo object
    """
    camera = idp.reconstruct.Photo()
    camera.id = int(xml_obj.attrib["id"])
    camera.sensor_id = int(xml_obj.attrib["sensor_id"])
    camera.label = xml_obj.attrib["label"]
    camera.orientation = int(xml_obj.findall("./orientation")[0].text)
    # get enabled for multispectral images
    # Exmaple of calibration_image group folder:
    """
    <camera id="932" sensor_id="0" label="DJI_20230901132303_0001_MS_G" enabled="false">
        <orientation>1</orientation>
        <reference x="..." y="..." z="..." yaw="..." pitch="..." roll="..." enabled="true" rotation_enabled="false"/>
    </camera>
    <camera id="935" sensor_id="3" master_id="932" label="DJI_20230901132303_0001_MS_NIR" enabled="false">
        <orientation>1</orientation>
    </camera>
    """
    # Example of common image:
    """
    <camera id="580" sensor_id="0" component_id="0" label="DJI_20230901130208_0001_MS_G">
      <transform>...</transform>
      <rotation_covariance>...</rotation_covariance>
      <location_covariance>...</location_covariance>
      <orientation>1</orientation>
      <reference x="..." y="..." z="..." yaw="..." pitch="..." roll="..." enabled="true" rotation_enabled="false"/>
    </camera>
    <camera id="583" sensor_id="3" component_id="0" master_id="580" label="DJI_20230901130208_0001_MS_NIR">
      <orientation>1</orientation>
    </camera>
    """
    if "enabled" not in xml_obj.attrib.keys():
        camera.enabled = True
    else:
        if xml_obj.attrib["enabled"] == 'false':
            camera.enabled = False
        else:
            camera.enabled = True

    if "master_id" in xml_obj.attrib.keys():
        camera.master_id = int(xml_obj.attrib["master_id"])

    # deal with camera have empty tags
    transform_tag = xml_obj.findall("./transform")
    if len(transform_tag) == 1:
        transform_str = transform_tag[0].text
        camera.transform = np.fromstring(transform_str, sep=" ", dtype=float).reshape((4, 4))
    else:
        if camera.master_id is None:
            # have no transform and not master_id, can not do the reverse caluclation
            camera.enabled = False

    shutter_rotation_tag = xml_obj.findall("./rolling_shutter/rotation")
    if len(shutter_rotation_tag) == 1:
        shutter_rotation_str = shutter_rotation_tag[0].text
        camera.rotation = np.fromstring(shutter_rotation_str, sep=" ", dtype=float).reshape((3, 3))

    shutter_translation_tag = xml_obj.findall("./rolling_shutter/translation")
    if len(shutter_translation_tag) == 1:
        shutter_translation_str = shutter_translation_tag[0].text
        camera.translation = np.fromstring(shutter_translation_str, sep=" ", dtype=float)

    return camera


def _decode_frame_xml(xml_str):
    """

    Parameters
    ----------
    xml_str: str
        the xml string from chunk.zip file, for example:

        <?xml version="1.0" encoding="UTF-8"?>
        <frame version="1.2.0">
          <cameras>
            <camera camera_id="0">
              <photo path="//172.31.12.56/pgg2020a/drone/20201029/goya/DJI_0284.JPG">
                <meta>
                  <property name="DJI/AbsoluteAltitude" value="+89.27"/>
                  <property name="DJI/FlightPitchDegree" value="+0.50"/>
                  <property name="DJI/FlightRollDegree" value="-0.40"/>
                  <property name="DJI/FlightYawDegree" value="+51.00"/>
                  <property name="DJI/GimbalPitchDegree" value="+0.00"/>
                  <property name="DJI/GimbalRollDegree" value="+0.00"/>
                  <property name="DJI/GimbalYawDegree" value="+0.30"/>
                  <property name="DJI/RelativeAltitude" value="+1.20"/>
                  <property name="Exif/ApertureValue" value="2.97"/>
                  <property name="Exif/DateTime" value="2020:10:29 14:13:09"/>
                  <property name="Exif/DateTimeOriginal" value="2020:10:29 14:13:09"/>
                  <property name="Exif/ExposureTime" value="0.001"/>
                  <property name="Exif/FNumber" value="2.8"/>
                  <property name="Exif/FocalLength" value="4.49"/>
                  <property name="Exif/FocalLengthIn35mmFilm" value="24"/>
                  <property name="Exif/GPSAltitude" value="89.27"/>
                  <property name="Exif/GPSLatitude" value="35.327145"/>
                  <property name="Exif/GPSLongitude" value="139.989069888889"/>
                  <property name="Exif/ISOSpeedRatings" value="200"/>
                  <property name="Exif/Make" value="DJI"/>
                  <property name="Exif/Model" value="FC7203"/>
                  <property name="Exif/Orientation" value="1"/>
                  <property name="Exif/ShutterSpeedValue" value="9.9657"/>
                  <property name="Exif/Software" value="v02.51.0008"/>
                  <property name="File/ImageHeight" value="3000"/>
                  <property name="File/ImageWidth" value="4000"/>
                  <property name="System/FileModifyDate" value="2020:10:29 14:13:10"/>
                  <property name="System/FileSize" value="5627366"/>
                </meta>
              </photo>
            </camera>
            ...
          </cameras>
          <markers>
            <marker marker_id="0">
              <location camera_id="230" pinned="true" x="3042.73071" y="1298.28467"/>
              <location camera_id="103" pinned="true" x="2838.39014" y="457.134155"/>
              ...
            </marker>
            ...
          </markers>
          <thumbnails path="thumbnails/thumbnails.zip"/>
          <point_cloud path="point_cloud/point_cloud.zip"/>
          <depth_maps id="0" path="depth_maps/depth_maps.zip"/>
          <dense_cloud id="0" path="dense_cloud/dense_cloud.zip"/>
        </frame>
    Returns
    -------

    """
    camera_meta = {}
    marker_meta = {}

    xml_tree = ElementTree.fromstring(xml_str)

    for camera_tag in xml_tree.findall("./cameras/camera"):
        camera_idx = int(camera_tag.attrib["camera_id"])
        # here need to resolve absolute path
        # <photo path="../../../../source/220613_G_M600pro/DSC06035.JPG">
        # this is the root to 220613_G_M600pro.files\0\0\frame.zip"
        # not the real path
        camera_path = camera_tag[0].attrib["path"]
        camera_meta[camera_idx] = camera_path

    return camera_meta, marker_meta


###############
# calculation #
###############

def apply_transform_matrix(points_xyz, matrix):
    """Transforms a point or points in homogeneous coordinates.
    equal to Metashape.Matrix.mulp() or Metashape.Matrix.mulv()

    Parameters
    ----------
    matrix: np.ndarray
        4x4 transform numpy array
    points_df: np.ndarray
        For example:

        .. code-block:: python

            # 1x3 single point
            >>> np.array([1,2,3])
               x  y  z
            0  1  2  3

            # nx3 points
            >>> np.array([[1,2,3], [4,5,6], ...])
               x  y  z
            0  1  2  3
            1  4  5  6
            ...

    Returns
    -------
    out: pd.DataFrame
        same size as input points_np:

        .. code-block:: text

               x  y  z
            0  1  2  3
            1  4  5  6

    """
    points_xyz, is_single = idp.geotools.is_single_point(points_xyz)

    point_ext = np.insert(points_xyz, 3, 1, axis=1)
    dot_matrix = point_ext.dot(matrix.T)
    dot_points = dot_matrix[:, 0:3] / dot_matrix[:, 3][:, np.newaxis]

    if is_single:
        return dot_points[0,:]
    else:
        return dot_points
