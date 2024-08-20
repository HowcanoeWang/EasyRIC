import re
import os
import oss2
import shutil
import pytest
import random
from pathlib import Path
from unittest.mock import patch, MagicMock

import easyidp as idp

def test_get_download_url_without_download():

    assert idp.data.Lotus.name == '2017_tanashi_lotus'
    assert idp.data.TestData.name == 'data_for_tests'

    url = idp.data.Lotus.gdrive_url
    assert isinstance(url, str)

def test_usr_data_dir():
    root_dir = idp.data.user_data_dir("")
    assert "easyidp.data" in str(root_dir)

def test_gdown():
    gd_dir = idp.data.user_data_dir("gdown_test")
    
    if gd_dir.exists():
        shutil.rmtree(gd_dir)
    
    gd = idp.data.GDownTest()

    assert gd.data_dir.exists()
    assert (gd.data_dir / "file1.txt").exists()

    assert Path(gd.pix4d.proj).resolve() == (gd.data_dir / "file1.txt").resolve()
    assert Path(gd.metashape.param).resolve() == (gd.data_dir / "folder1").resolve()

    # test remove and reload
    gd.remove_data()
    assert not gd.data_dir.exists()

    gd.reload_data()
    assert gd.data_dir.exists()

#====================
# AliYun Downloading 
#====================

ali_down = idp.data.AliYunDownloader()

def test_class_aliyun_downloader_init():

    assert ali_down.bucket_name == "easyidp-data"
    assert isinstance(ali_down.bucket, oss2.api.Bucket)

def test_class_aliyun_downloader_cost():
    # test cost calculate
    cost = ali_down.calculate_download_cost("aaa", "5.6GB")
    assert cost >= (0.12 + 0.5) * 5.6
    assert cost <= (0.12 + 0.5 + 0.1) * 5.6

    # test different size
    cost = ali_down.calculate_download_cost("bbb", "5.6MB")
    cost = ali_down.calculate_download_cost("ccc", "5.6KB")

    # test incorrect string
    with pytest.raises(ValueError, match=re.escape("Invalid dataset size format of ddd.size = 5.6.7MB")):
        cost = ali_down.calculate_download_cost("ddd", "5.6.7MB")

    # test unsupported file size
    with pytest.raises(ValueError, match=re.escape("Invalid dataset size format of eee.size = 5.6TB")):
        cost = ali_down.calculate_download_cost("eee", "5.6TB")

#-----------------
# download_auth()
#-----------------
def test_class_aliyun_downloader_auth_success():
    random.seed(10)
    dataset_name = "aaa"
    dataset_size = "0.5GB"

    money_cost = ali_down.calculate_download_cost(dataset_name, dataset_size)
    confirm_str = f"我已知悉此次下载会消耗{money_cost}元的下行流量费用"

    with patch('builtins.input', side_effect=[confirm_str]):
        random.seed(10)
        assert ali_down.download_auth(dataset_name, dataset_size) == True

def test_class_aliyun_downloader_auth_wrong():
    random.seed(10)
    dataset_name = "bbb"
    dataset_size = "0.5GB"

    wrong_inputs = ["wrong input"] * 5

    with patch('builtins.input', side_effect=wrong_inputs):
        with pytest.raises(PermissionError):
            ali_down.download_auth(dataset_name, dataset_size)

def test_download_auth_retry_then_success():
    random.seed(10)
    dataset_name = "ccc"
    dataset_size = "0.5GB"

    # 模拟前几次失败，然后成功的用户输入
    money_cost = ali_down.calculate_download_cost(dataset_name, dataset_size)
    confirm_str = f"我已知悉此次下载会消耗{money_cost}元的下行流量费用"
    inputs = ["wrong input", "wrong input", confirm_str]

    with patch('builtins.input', side_effect=inputs):
        random.seed(10)
        assert ali_down.download_auth(dataset_name, dataset_size) == True

#------------
# download()
#------------
"""
在这个测试文件中，我们定义了以下内容：

+ mock_requests_get：一个 pytest fixture，用于模拟 requests.get 请求。
- mock_oss2：一个 pytest fixture，用于模拟 oss2.Bucket 和 oss2.resumable_download。
+ test_download_success：测试 download 方法在成功下载时的行为。
+ test_download_auth_failure：测试在获取认证失败时的行为。

运行这些测试时，patch 会替换 requests.get 和 oss2 模块中的相关函数，使其返回预定义的值，从而避免实际的网络请求和下载操作。
"""
@pytest.fixture
def mock_requests_get():
    with patch('requests.get') as mock_get:
        yield mock_get

def test_download_success():
    dataset_name = "gdown_test"
    output = "./tests/out/data_test/gdown_download_test.zip"

    if os.path.exists(output):
        os.remove(output)

    ali_down.download(dataset_name, output)

    assert os.path.exists(output)

def test_download_auth_failure(mock_requests_get):
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_requests_get.return_value = mock_response

    with pytest.raises(ConnectionRefusedError):
        idp.data.AliYunDownloader()

#--------------------
# GDownDataset class
#--------------------

def test_gdown_ali_oss():
    # the default EasyIDPDataset using system cache folder
    # it conflits when pytest parallel 
    #  -> previous `gdown_test` function may delete the same folder
    # thus made an unique test, which can specify unzip folder 
    from easyidp.data import EasyidpDataSet, GDOWN_TEST_URL
    class ADownTest(EasyidpDataSet):
        def __init__(self, dataset_folder:Path):

            super().__init__("gdown_test", GDOWN_TEST_URL, "0.2KB")
            self.data_dir = dataset_folder / self.name
            self.zip_file = dataset_folder / (self.name + ".zip")

            super().load_data()

            self.pix4d.proj = self.data_dir / "file1.txt"
            self.metashape.param = self.data_dir / "folder1"
    # end of unique testing class

    dataset_folder = Path('./tests/out/data_test/')
    data_dir = dataset_folder / "gdown_test"
    
    # clear already existed folder for `gdown_test`
    if data_dir.exists():
        shutil.rmtree(data_dir)
    
    # ask if China mainland, answer: no
    #   google drive not available, and not in china mainland
    #   => not provide aliyun oss service for overseas, 
    #      => notice google drive download link broken, report to github
    inputs = ["n"]
    with patch('builtins.input', side_effect=inputs):
        with pytest.raises(
            ConnectionError, 
            match=re.escape(
                "Could not find proper downloadable link for dataset gdown_test."
            )
            ):
            # force to use AliYUN OSS
            idp.GOOGLE_AVAILABLE = False
            gd = ADownTest(dataset_folder=dataset_folder)

    # ask if China mainland, answer: yes
    inputs = ["y", "我已知悉此次下载会消耗0.0元的下行流量费用"]
    with patch('builtins.input', side_effect=inputs):
        # force to use AliYUN OSS
        idp.GOOGLE_AVAILABLE = False
        gd = ADownTest(dataset_folder=dataset_folder)

    assert gd.data_dir.exists()
    assert (gd.data_dir / "file1.txt").exists()