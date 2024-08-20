import os
import sys
import shutil
import zipfile
import requests
import subprocess
import tqdm
from pathlib import Path

GDOWN_TEST_URL = "https://drive.google.com/file/d/1yWvIOYJ1ML-UGleh3gT5b7dxXzBuSPgQ/view?usp=share_link"

def user_data_dir(file_name=""):
    r"""Get OS specific data directory path for EasyIDP.
    
    Parameters
    ----------
    file_name : str
        file to be fetched from the data dir

    Returns
    -------
    str
        full path to the user-specific data dir

    Notes
    -----
    Typical user data directories are:

    .. code-block:: text

        macOS:    ~/Library/Application Support/easyidp.data
        Unix:     ~/.local/share/easyidp.data   # or in $XDG_DATA_HOME, if defined
        Win 10:   C:\Users\<username>\AppData\Local\easyidp.data

    For Unix, we follow the XDG spec and support ``$XDG_DATA_HOME`` if defined.

    Referenced from stackoverflow [1]_ then get github [2]_ .

    References
    ----------
    .. [1] Python: Getting AppData folder in a cross-platform way https://stackoverflow.com/questions/19078969/python-getting-appdata-folder-in-a-cross-platform-way
    .. [2] SwagLyrics-For-Spotify/swaglyrics/__init__.py https://github.com/SwagLyrics/SwagLyrics-For-Spotify/blob/master/swaglyrics/__init__.py#L8-L32

    """
    # get os specific path
    if sys.platform.startswith("win"):
        os_path = os.getenv("LOCALAPPDATA")
    elif sys.platform.startswith("darwin"):
        os_path = "~/Library/Application Support"
    else:
        # linux
        os_path = os.getenv("XDG_DATA_HOME", "~/.local/share")

    # join with easyidp.data dir
    path = Path(os_path) / "easyidp.data"

    add_usr = path.expanduser()

    if not os.path.exists(str(add_usr)):
        os.mkdir(str(add_usr))

    return add_usr / file_name

def show_data_dir():
    """open the cached data files in cross-platform system default viewer.

    Notes
    -----
    It modified from this [1]_ webpage.

    References
    ----------
    .. [1] Python: Opening a folder in Explorer/Nautilus/Finder https://stackoverflow.com/questions/6631299/python-opening-a-folder-in-explorer-nautilus-finder

    """
    path = user_data_dir()

    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform.startswith("darwin"):
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def url_checker(url):
    r"""Check if download url is accessable or not.
    
    Modified from this [1]_ link.

    Parameters
    ----------
    url : str
        The url need to be checked

    References
    ----------
    .. [1] https://pytutorial.com/check-url-is-reachable
    """
    try:
        #Get Url
        get = requests.get(url, timeout=3)
        # if the request succeeds 
        if get.status_code == 200:
            return True
        else:
            return False

    #Exception
    except requests.exceptions.RequestException as e:
        # print URL with Errs
        return False

def _can_access_google_cloud():
    return url_checker(GDOWN_TEST_URL)


def download_all():
    """download all datasets
    """
    lotus = Lotus()
    gd = GDownTest()
    test = TestData()


class AliYunDownloader():

    def __init__(self):
        import oss2

        access_url = "https://easyidp-data.oss-rg-china-mainland.aliyuncs.com/access.txt"

        response = requests.get(access_url)

        self.tqdm_bar = None

        if response.status_code == 200:
            content = response.text
            access_key_secret, access_key_id = content.split('\r\n')

            self.bucket_name = 'easyidp-data'
            self.endpoint = 'oss-rg-china-mainland.aliyuncs.com'

            self.bucket = oss2.Bucket(
                oss2.Auth(access_key_id, access_key_secret), 
                self.endpoint, self.bucket_name
            )
        else:
            raise ConnectionRefusedError(
                f"Failed to achieve AliYun Auth token, Please contact the maintainer via github.\n"
                f"Status code: {response.status_code}"
            )
        
    @staticmethod
    def calculate_download_cost(dataset_name, dataset_size):

        import re
        import random

        # Define the cost per GB
        cost_per_gb = 0.12 + 0.5 + 0.1*random.random()
        
        # Regular expression to match the number and unit
        pattern = re.compile(r'(\d+(\.\d+)?)([KMG]B)')
        match = pattern.match(dataset_size)
        
        if not match:
            raise ValueError(f"Invalid dataset size format of {dataset_name}.size = {dataset_size}")
        
        # Extract the value and unit
        value = float(match.group(1))
        unit = match.group(3)
        
        # Convert the value to GB
        if unit == 'KB':
            value_in_gb = value / (1024 * 1024)
        elif unit == 'MB':
            value_in_gb = value / 1024
        elif unit == 'GB':
            value_in_gb = value
        else:
            raise ValueError(f"Unsupported dataset size unit {dataset_size}")
        
        # Calculate the cost
        cost = value_in_gb * cost_per_gb
        return round(cost, 1)
    
    def download_auth(self, dataset_name, dataset_size):
        money_cost = self.calculate_download_cost(dataset_name, dataset_size)
        confirm_str = f"我已知悉此次下载会消耗{money_cost}元的下行流量费用"

        # 插入不可见字符
        invis_char = '\u200B'  # 零宽度空格
        no_copy_confirm_str = confirm_str[0:10] + invis_char + confirm_str[10:20] + invis_char + confirm_str[20:]

        # 使用ANSI转义序列设置颜色和格式
        RED = '\033[91m'
        BOLD = '\033[1m'
        END = '\033[0m'

        notification = (
            f"{RED}{BOLD}请注意，中国大陆数据集下载使用作者私人搭建的阿里云文件存储服务，\n"
            f"下载数据集会产生一定的流量费用(下载当前数据集{dataset_name}会消耗大约{money_cost}元)，\n"
            f"此费用由作者本人负担，请勿在非必要的情况下重复下载或将此数据存储仓库用于其他用途\n\n"
            f"如果同意以上内容，请在下方用输入法输入（复制无效)：\n{no_copy_confirm_str}{END}"
        )
        print(notification)

        retry_counter = 0
        matched = False

        while retry_counter < 5:
            user_confirm = input(">>> ")
            if user_confirm == confirm_str:
                matched = True
                break
            else:
                print("输入的字符不匹配，请用输入法再次输入\n")
                retry_counter += 1

        if not matched:
            raise PermissionError("尝试失败次数超过5次，请重新运行脚本")
        
        return matched
        
    def tqdm_progress_bar(self, consumed_bytes, total_bytes):
        if total_bytes:
            if self.tqdm_bar is None:
                # 创建 tqdm 进度条实例
                self.tqdm_bar = tqdm.tqdm(total=total_bytes, unit='B', unit_scale=True, desc='Downloading from Aliyun OSS')

            # rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
            # 使用tqdm显示进度条
            if self.tqdm_bar:
                self.tqdm_bar.n = consumed_bytes
                self.tqdm_bar.refresh()

            # sys.stdout.flush()

    def download(self, dataset_name, output):
        import oss2

        self.tqdm_bar = None
        oss2.resumable_download(self.bucket, dataset_name+'.zip', output, 
                                multiget_threshold=30000*1024,
                                part_size=10000*1024,
                                progress_callback=self.tqdm_progress_bar)

        # 下载完成后关闭进度条
        if self.tqdm_bar:
            self.tqdm_bar.close()


class EasyidpDataSet():
    """The base class for Dataset
    """

    def __init__(self, name="", gdrive_url="", size=""):
        """The dataset has the following properties (almost in string type)

        name
            The dataset name
        gdrive_url
            The google drive download urls
        size
            Ths size of zipped file
        data_dir
            The final dataset directory
        zip_file
            The *temporary* downloaded zip file path (won't be used in)
        shp
            The path to plot ROI shapefile (\*.shp)
        photo
            The folder path to the raw photos

        pix4d.proj
            The pix4d project folder
        pix4d.param
            The parameter folder of pix4d porject
        pix4d.dom
            The generated DOM path of plot map
        pix4d.dsm
            The generated DSM path of plot map
        pix4d.pcd
            The generated pointcloud path of plot map

        metashape.proj
            The metashape project file
        metashape.param
            The parameter folder of metashape porject
        metashape.dom
            The generated DOM path of plot map
        metashape.dsm
            The generated DSM path of plot map
        metashape.pcd
            The generated pointcloud path of plot map
        

        """
        self.name = name
        self.gdrive_url = gdrive_url
        self.size = size
        self.data_dir = user_data_dir(self.name)
        self.zip_file = user_data_dir(self.name + ".zip")

        self.pix4d = self.ReconsProj()
        self.metashape = self.ReconsProj()

    def load_data(self):
        r"""Download dataset from Google Drive to user AppData folder
        """

        if not os.path.exists(self.data_dir):
            
            if not os.path.exists(self.zip_file):
                self._download_data()

            if os.path.exists(self.zip_file):
                print("Successfully downloaded, start unzipping ...")
                self._unzip_data()
                print("Successfully unzipped, the cache zip file has been removed.")
            else:
                raise FileNotFoundError(
                    f"Could not find the downloaded file [{self.zip_file}], "
                    f"please call ().load_data() to download again.\n"
                    f"Tips: ensure you can access Google Drive url."
                )

    def reload_data(self):
        """remove local data and redownload again
        """
        self.remove_data()
        self.load_data()

    def remove_data(self):
        """remove local cached data file
        """
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        
    def _download_data(self):
        """using gdown to download dataset from Google Drive to user AppData folder
        """
        # Download; extract data to disk.
        # Raise an exception if the link is bad, or we can't connect, etc.
        import easyidp as idp

        if idp.GOOGLE_AVAILABLE:
            import gdown
            # google drive gdown_test.zip file is accessable
            # then try according google drive download link
            if url_checker(self.gdrive_url):
                output = gdown.download(url=self.gdrive_url, output=str(self.zip_file), quiet=False, fuzzy=True)
            else:
                # user can access Google Drive but maybe dataset zip file is missing, no waste AliYun OSS resource
                # just mention user to double check google drive access
                raise ConnectionError(
                    f"Could not access google download link for dataset {self.name} from: \n{self.gdrive_url}.\n"
                    f"Please contact the maintainer via github if above link is broken."
                )
        else:
            # high possibility in China Mainland, use aliyun OSS for downloading
            is_mainland_user = input("Google Drive Unaccessable, are you locate in China Mainland? (Y/N)\n>>> ")
            if is_mainland_user in ["Yes", "Y", "y", "yes"]:
                if idp.aliyun_down is None:
                    idp.aliyun_down = AliYunDownloader()

                idp.aliyun_down.download_auth(dataset_name=self.name, dataset_size=self.size)
                idp.aliyun_down.download(dataset_name=self.name, output=str(self.zip_file))
            else:
                raise ConnectionError(
                    f"Could not find proper downloadable link for dataset {self.name}.\n"
                    f"Please contact the maintainer via github.")

    def _unzip_data(self):
        """Unzip downloaded zip data and remove after decompression
        """
        with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)

        # already extracted
        if os.path.exists(self.data_dir):
            os.remove(self.zip_file)
        else:
            raise FileNotFoundError("Seems fail to unzip, please check whether the zip file is fully downloaded.")
        

    class ReconsProj():


        def __init__(self) -> None:
            self.project = ""
            self.param = ""
            self.dom = ""
            self.dsm = ""
            self.pcd = ""


class Lotus(EasyidpDataSet):
    """The dataset for lotus plot in Tanashi, Tokyo.

    .. image:: ../../_static/images/data/2017_tanashi_lotus.png 
        :width: 600
        :alt: 2017_tanashi_lotus.png 

    - **Crop** : lotus
    - **Location** : Tanashi, Nishi-Tokyo, Japan
    - **Flight date** : May 31, 2017
    - **UAV model** : DJI Inspire 1
    - **Flight height** : 30m
    - **Image number** :142
    - **Image size** : 4608 x 3456
    - **Software** : Pix4D, Metashape
    - **Outputs** : DOM, DSM, PCD
    """

    name = "2017_tanashi_lotus"
    size = "3.3GB"
    gdrive_url = "https://drive.google.com/file/d/1SJmp-bG5SZrwdeJL-RnnljM2XmMNMF0j/view?usp=share_link"

    def __init__(self):
        """
        Containts the following arguments, you can access by:

        .. code-block:: python

            >>> lotus = idp.data.Lotus()
            >>> lotus.photo
            'C:\\Users\\<user>\\AppData\\Local\\easyidp.data\\2017_tanashi_lotus\\20170531\\photos'

        - ``.photo`` : the folder containts raw images
        - ``.shp`` : the plot roi shapefile
        - ``.pix4d.project`` : the pix4d project folder
        - ``.pix4d.param`` : the pix4d project parameter folder
        - ``.pix4d.dom`` : the pix4d produced orthomosaic
        - ``.pix4d.dsm`` : the pix4d produced digial surface del
        - ``.pix4d.pcd`` : the pix4d produced point cloud
        - ``.metashape.project`` : the metashape project file
        - ``.metashape.param`` : the metashape project folder
        - ``.metashape.dom`` : the metashape produced orthomosaic
        - ``.metashape.pcd`` : the metashape produced point cloud
        - ``.metashape.dsm`` : the metashape produced digial surface model

        See also
        --------
        EasyidpDataSet
        """
        self.data_dir = user_data_dir(self.name)
        self.zip_file = user_data_dir(self.name + ".zip")

        self.pix4d = self.ReconsProj()
        self.metashape = self.ReconsProj()

        super().load_data()

        self.photo = self.data_dir / "20170531" / "photos"
        self.shp = self.data_dir / "plots.shp"

        self.pix4d.project = self.data_dir / "20170531"
        self.pix4d.param = self.data_dir / "20170531" / "params"
        
        self.pix4d.dom = self.data_dir / "20170531" / "hasu_tanashi_20170531_Ins1RGB_30m_transparent_mosaic_group1.tif"
        self.pix4d.dsm = self.data_dir / "20170531" / "hasu_tanashi_20170531_Ins1RGB_30m_dsm.tif"
        self.pix4d.pcd = self.data_dir / "20170531" / "hasu_tanashi_20170531_Ins1RGB_30m_group1_densified_point_cloud.ply"

        self.metashape.project = self.data_dir / "170531.Lotus.psx"
        self.metashape.param = self.data_dir / "170531.Lotus.files"
        self.metashape.dom = self.data_dir / "170531.Lotus.outputs" / "170531.Lotus_dom.tif"
        self.metashape.dsm = self.data_dir / "170531.Lotus.outputs" / "170531.Lotus_dsm.tif"
        self.metashape.pcd = self.data_dir / "170531.Lotus.outputs" / "170531.Lotus.laz"

class ForestBirds(EasyidpDataSet):
    """The dataset for forest ecology survey, provided by the University of Florida.

    .. image:: ../../_static/images/data/2022_florida_forestbirds.png 
        :width: 600
        :alt: 2022_florida_forestbird.png 

    - **Author**: Prof. Ben Weinstein, The University of Florida.
    - **Location** : Florida, US
    - **Flight date** : March 24, 2022
    - **UAV model** : DJI FC6540
    - **Flight height** : ?? m
    - **Image number** :93
    - **Image size** : 6016 x 4008
    - **Software** : Metashape
    - **Outputs** : DOM, DSM, PCD
    """

    name = "2022_florida_forestbirds"
    size = "1.97GB"
    gdrive_url = "https://drive.google.com/file/d/1mXkzaoSSCAA87cxcMHKL6_VNlykRYxJr/view?usp=sharing"

    def __init__(self):
        """
        Containts the following arguments, you can access by:

        .. code-block:: python

            >>> fb = idp.data.ForestBirds()
            >>> fb.photo
            'C:\\Users\\<user>\\AppData\\Local\\easyidp.data\\2022_florida_forestbirds\\Hidden_Little_03_24_2022'

        - ``.photo`` : the folder containts raw images
        - ``.shp`` : the plot roi shapefile
        - ``.metashape.project`` : the metashape project file
        - ``.metashape.param`` : the metashape project folder
        - ``.metashape.dom`` : the metashape produced orthomosaic
        - ``.metashape.dsm`` : the metashape produced digial surface model

        See also
        --------
        EasyidpDataSet
        """
        self.data_dir = user_data_dir(self.name)
        self.zip_file = user_data_dir(self.name + ".zip")

        self.metashape = self.ReconsProj()

        super().load_data()

        self.photo = self.data_dir / "Hidden_Little_03_24_2022"
        self.shp = self.data_dir / "Hidden_Little_grid.shp"

        self.metashape.project = self.data_dir / "Hidden_Little_03_24_2022.psx"
        self.metashape.param = self.data_dir / "Hidden_Little_03_24_2022.files"
        self.metashape.dom = self.data_dir / "Hidden_Little_03_24_2022.tiff"
        self.metashape.dsm = self.data_dir / "Hidden_Little_03_24_2022_DEM.tif"
        
class GDownTest(EasyidpDataSet):
    """The data for Google Drive download testing
    """

    def __init__(self):

        super().__init__("gdown_test", GDOWN_TEST_URL, "0.2KB")
        super().load_data()

        self.pix4d.proj = self.data_dir / "file1.txt"
        self.metashape.param = self.data_dir / "folder1"


class TestData(EasyidpDataSet):
    """The data for developer and package testing.
    """

    name = "data_for_tests"
    size = "344MB"
    gdrive_url = "https://drive.google.com/file/d/17b_17CofqIuCVOWMnD67_wOnWMtwF8bw/view?usp=share_link"

    def __init__(self, test_out="./tests/out"):
        """
        Containts the following arguments, you can access by:
        
        **json test module** 

        * ``.json.for_read_json``
        * ``.json.labelme_demo``
        * ``.json.labelme_warn``
        * ``.json.labelme_err``

        **shp test module** 

        * ``.shp.lotus_shp`` 
        * ``.shp.lotus_prj`` 
        * ``.shp.complex_shp``
        * ``.shp.complex_prj``
        * ``.shp.lonlat_shp``
        * ``.shp.utm53n_shp``
        * ``.shp.utm53n_prj``
        * ``.shp.rice_shp``
        * ``.shp.rice_prj``
        * ``.shp.roi_shp``
        * ``.shp.roi_prj``
        * ``.shp.testutm_shp``
        * ``.shp.testutm_prj``

        **pcd test module** 

        * ``.pcd.lotus_las``
        * ``.pcd.lotus_laz``
        * ``.pcd.lotus_pcd``
        * ``.pcd.lotus_las13``
        * ``.pcd.lotus_laz13``
        * ``.pcd.lotus_ply_asc``
        * ``.pcd.lotus_ply_bin``
        * ``.pcd.maize_las``
        * ``.pcd.maize_laz``
        * ``.pcd.maize_ply``

        **roi test module** 

        * ``.roi.dxf``
        * ``.roi.lxyz_txt``
        * ``.roi.xyz_txt``

        **geotiff test module**

        * ``.tiff.soyweed_part``
        * ``.tiff.out``

        **metashape test module** 

        * ``.metashape.goya_psx``
        * ``.metashape.goya_param``
        * ``.metashape.lotus_psx``
        * ``.metashape.lotus_param``
        * ``.metashape.lotus_dsm``
        * ``.metashape.wheat_psx``
        * ``.metashape.wheat_param``
        * ``.metashape.multichunk_psx``
        * ``.metashape.multichunk_param``
        * ``.metashape.two_calib``
        * ``.metashape.multi_spectral``


        **pix4d test module** 

        * ``.pix4d.lotus_folder``
        * ``.pix4d.lotus_param``
        * ``.pix4d.lotus_photos``
        * ``.pix4d.lotus_dom``
        * ``.pix4d.lotus_dsm``
        * ``.pix4d.lotus_pcd``
        * ``.pix4d.lotus_dom_part``
        * ``.pix4d.lotus_dsm_part``
        * ``.pix4d.lotus_pcd_part``
        * ``.pix4d.maize_folder``
        * ``.pix4d.maize_dom``
        * ``.pix4d.maize_dsm``
        * ``.pix4d.maize_noparam``
        * ``.pix4d.maize_empty``
        * ``.pix4d.maize_noout``

        **cvtools test module** 

        * ``.cv.out``
        
        **visualize test module** 

        * ``.vis.out``


        Parameters
        ----------
        test_out : str, optional
            The folder for saving temporary outputs, by default "./tests/out"

        See also
        --------
        EasyidpDataSet
        """
        self.data_dir = user_data_dir(self.name)
        self.zip_file = user_data_dir(self.name + ".zip")

        super().load_data()

        self.json = self.JsonDataset(self.data_dir, test_out)
        self.metashape = self.MetashapeDataset(self.data_dir, test_out)
        self.pix4d = self.Pix4Dataset(self.data_dir)
        self.shp = self.ShapefileDataset(self.data_dir, test_out)
        self.pcd = self.PointCloudDataset(self.data_dir, test_out)
        self.roi = self.ROIDataset(self.data_dir)
        self.tiff = self.TiffDataSet(self.data_dir, test_out)

        self.cv = self.CVDataset(self.data_dir, test_out)
        self.vis = self.VisualDataset(self.data_dir, test_out)
        self.b2r = self.Back2rawDataset(self.data_dir, test_out)


    class MetashapeDataset():

        def __init__(self, data_dir, test_out):
            if isinstance(test_out, str):
                test_out = Path(test_out)

            self.goya_psx    = data_dir / "metashape" / "goya_test.psx"
            self.goya_param  = data_dir / "metashape" / "goya_test.files"

            self.lotus_psx   = data_dir / "metashape" / "Lotus.psx"
            self.lotus_param = data_dir / "metashape" / "Lotus.files"
            self.lotus_dsm   = data_dir / "metashape" / "Lotus.files" / "170531.Lotus_dsm.tif"

            self.wheat_psx   = data_dir / "metashape" / "wheat_tanashi.psx"
            self.wheat_param = data_dir / "metashape" / "wheat_tanashi.files"

            self.multichunk_psx   = data_dir / "metashape" / "multichunk.psx"
            self.multichunk_param = data_dir / "metashape" / "multichunk.files"

            self.multifolder_psx   = data_dir / "metashape" / "multifolder.psx"
            self.multifolder_param = data_dir / "metashape" / "multifolder.files"

            self.nestedfolder_psx   = data_dir / "metashape" / "nestedfolders.psx"
            self.nestedfolder_param = data_dir / "metashape" / "nestedfolders.files"

            self.camera_disorder_psx   = data_dir / "metashape" / "camera_disorder.psx"
            self.camera_disorder_param = data_dir / "metashape" / "camera_disorder.files"

            self.two_calib_psx   = data_dir / "metashape" / "two_calib.psx"
            self.two_calib_param = data_dir / "metashape" / "two_calib.files"

            self.multi_spectral_psx   = data_dir / "metashape" / "multi_spectral.psx"
            self.multi_spectral_param = data_dir / "metashape" / "multi_spectral.files"


    class Pix4Dataset():

        def __init__(self, data_dir):

            # here is reorgainzed pix4d project
            self.lotus_folder   = data_dir / "pix4d" / "lotus_tanashi_full"
            self.lotus_param    = self.lotus_folder / "params"
            self.lotus_photos   = self.lotus_folder / "photos"
            self.lotus_dom      = self.lotus_folder / "hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif"
            self.lotus_dsm      = self.lotus_folder / "hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif"
            self.lotus_pcd      = self.lotus_folder / "hasu_tanashi_20170525_Ins1RGB_30m_group1_densified_point_cloud.ply"
            self.lotus_dom_part = self.lotus_folder / "plot_dom.tif"
            self.lotus_dsm_part = self.lotus_folder / "plot_dsm.tif"
            self.lotus_pcd_part = self.lotus_folder / "plot_pcd.ply"

            # here is standard pix4d project
            self.maize_folder  = data_dir / "pix4d" / "maize_tanashi" / "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d"
            self.maize_dom = self.maize_folder / "3_dsm_ortho" / "2_mosaic" / \
                "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_transparent_mosaic_group1.tif"
            self.maize_dsm = self.maize_folder / "3_dsm_ortho" / "1_dsm" / \
                "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_dsm.tif"

            self.maize_noparam = data_dir / "pix4d" / "maize_tanashi" / "maize_tanashi_no_param"
            self.maize_empty   = data_dir / "pix4d" / "maize_tanashi" / "maize_tanashi_raname_empty_test"
            self.maize_noout   = data_dir / "pix4d" / "maize_tanashi" / "maize_tanashi_raname_no_outputs"

    class JsonDataset():

        def __init__(self, data_dir, test_out):
            self.data_dir = data_dir
            self.for_read_json = data_dir / "json_test" / "for_read_json.json"
            self.labelme_demo  = data_dir / "json_test" / "labelme_demo_img.json"
            self.labelme_warn  = data_dir / "json_test" / "labelme_warn_img.json"
            self.labelme_err   = data_dir / "json_test" / "for_read_json.json"

            self.geojson_soy   = data_dir / "json_test" / "2023_soybean_field.geojson"

            if isinstance(test_out, str):
                test_out = Path(test_out)
            self.out = test_out / "json_test"

        def __truediv__(self, other):
            return self.data_dir / "json_test" / other


    class ShapefileDataset():

        def __init__(self, data_dir, test_out):

            self.data_dir = data_dir
            self.lotus_shp = data_dir / "shp_test" / "lotus_plots.shp"
            self.lotus_prj = data_dir / "shp_test" / "lotus_plots.prj"

            self.complex_shp = data_dir / "shp_test" / "complex_shp_review.shp"
            self.complex_prj = data_dir / "shp_test" / "complex_shp_review.prj"

            self.lonlat_shp = data_dir / "shp_test" / "lon_lat.shp"

            self.utm53n_shp = data_dir / "shp_test" / "lon_lat_utm53n.shp"
            self.utm53n_prj = data_dir / "shp_test" / "lon_lat_utm53n.prj"

            self.rice_shp = data_dir / "shp_test" / "rice_ind_duplicate.shp"
            self.rice_prj = data_dir / "shp_test" / "rice_ind_duplicate.prj"

            self.roi_shp = data_dir / "shp_test" / "roi.shp"
            self.roi_prj = data_dir / "shp_test" / "roi.prj"

            self.testutm_shp = data_dir / "shp_test" / "test_utm.shp"
            self.testutm_prj = data_dir / "shp_test" / "test_utm.prj"

            self.jp_crs_shp = data_dir / "shp_test" / "jp_crs.shp"
            self.jp_crs_prj = data_dir / "shp_test" / "jp_crs.prj"

            if isinstance(test_out, str):
                test_out = Path(test_out)
            self.out = test_out / "shp_test"

        def __truediv__(self, other):
            return self.data_dir / "shp_test" / other


    class PointCloudDataset():

        def __init__(self, data_dir, test_out):
            self.data_dir = data_dir

            self.lotus_las = data_dir / "pcd_test" / "hasu_tanashi.las"
            self.lotus_laz = data_dir / "pcd_test" / "hasu_tanashi.laz"
            self.lotus_pcd = data_dir / "pcd_test" / "hasu_tanashi.pcd"

            self.lotus_las13 = data_dir / "pcd_test" / "hasu_tanashi_1.3.las"
            self.lotus_laz13 = data_dir / "pcd_test" / "hasu_tanashi_1.3.laz"

            self.lotus_ply_asc = data_dir / "pcd_test" / "hasu_tanashi_ascii.ply"
            self.lotus_ply_bin = data_dir / "pcd_test" / "hasu_tanashi_binary.ply"

            self.maize_las = data_dir / "pcd_test" / "maize3na_20210614_15m_utm.las"
            self.maize_laz = data_dir / "pcd_test" / "maize3na_20210614_15m_utm.laz"
            self.maize_ply = data_dir / "pcd_test" / "maize3na_20210614_15m_utm.ply"

            if isinstance(test_out, str):
                test_out = Path(test_out)
            self.out = test_out / "pcd_test"

        def __truediv__(self, other):
            return self.data_dir / "pcd_test" / other


    class ROIDataset():

        def __init__(self, data_dir):
            self.data_dir = data_dir

            self.dxf  = data_dir / "roi_test" / "hasu_tanashi_ccroi.dxf"
            self.lxyz_txt = data_dir / "roi_test" / "hasu_tanashi_lxyz.txt"
            self.xyz_txt  = data_dir / "roi_test" / "hasu_tanashi_xyz.txt"


    class TiffDataSet():

        def __init__(self, data_dir, test_out):
            self.data_dir = data_dir

            self.soyweed_part = data_dir / "tiff_test" / "2_12.tif"

            if isinstance(test_out, str):
                test_out = Path(test_out)
            self.out = test_out / "tiff_test"

        def __truediv__(self, other):
            return self.data_dir / "tiff_test" / other


    class CVDataset():

        def __init__(self, data_dir, test_out):
            self.data_dir = data_dir

            if isinstance(test_out, str):
                test_out = Path(test_out)
            self.out = test_out / "cv_test"

        def __truediv__(self, other):
            return self.data_dir / "cv_test" / other


    class VisualDataset():

        def __init__(self, data_dir, test_out):
            self.data_dir = data_dir

            if isinstance(test_out, str):
                test_out = Path(test_out)
            self.out = test_out / "visual_test"

        def __truediv__(self, other):
            return self.data_dir / "visual_test" / other
        


    class Back2rawDataset():

        def __init__(self, data_dir, test_out):
            self.data_dir = data_dir

            if isinstance(test_out, str):
                test_out = Path(test_out)
            self.out = test_out / "back2raw_test"

        def __truediv__(self, other):
            return self.data_dir / "back2raw_test" / other
