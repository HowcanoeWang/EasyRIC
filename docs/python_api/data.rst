============
Data
============

.. currentmodule:: easyidp.data

Dataset
=======

The datasets are as follows (**for user**):

.. autosummary::
    :toctree: autodoc

    Lotus
    ForestBirds

Use example:

.. code-block:: python

    >>> import easyidp as idp
    >>> lotus = idp.data.Lotus()
    Downloading...
    From: https://drive.google.com/uc?id=1SJmp-bG5SZrwdeJL-RnnljM2XmMNMF0j
    To: C:\Users\<user>\AppData\Local\easyidp.data\2017_tanashi_lotus.zip
    100%|█████████████████████████████| 3.58G/3.58G [00:54<00:00, 65.4MB/s]
    >>> lotus.shp
    'C:\\Users\\<user>\\AppData\\Local\\easyidp.data\\2017_tanashi_lotus\\plots.shp'
    >>> lotus.metashape.proj
    'C:\\Users\\<user>\\AppData\\Local\\easyidp.data\\2017_tanashi_lotus\\170531.Lotus.psx'
    >>> lotus.photo
    'C:\\Users\\<user>\\AppData\\Local\\easyidp.data\\2017_tanashi_lotus\\20170531\\photos'
    >>> lotus.pix4d.param
    'C:\\Users\\<user>\\AppData\\Local\\easyidp.data\\2017_tanashi_lotus\\20170531\\params'

.. caution::

    For Chinese mainland user who can not access the GoogleDrive directly, the author personally setup an AliYun OSS downloading services in **2.0.1**, please update EasyIDP to the latest version (probably required :ref:`using-from-source-code`) and try:

    .. code-block:: python

        >>> lotus = idp.data.Lotus()
        Google Drive Unaccessable, are you locate in China Mainland? (Y/N)
        >>> Y

    Then it will give an cost notice in Chinese for you to confirm:

    .. code-block:: text

        请注意，中国大陆数据集下载使用作者私人搭建的阿里云文件存储服务，
        下载数据集会产生一定的流量费用(下载当前数据集2017_tanashi_lotus会消耗大约2.1元)，
        此费用由作者本人负担，请勿在非必要的情况下重复下载或将此数据存储仓库用于其他用途

        如果同意以上内容，请在下方用输入法输入（复制无效)：
        我已知悉此次下载会消耗2.1元的下行流量费用

    After typing the required sentences, it will downloading the dataset automatically from AliYun OSS services just like google cloud performance.

    .. code-block:: python

        >>> 我已知悉此次下载会消耗2.1元的下行流量费用
        Downloading from Aliyun OSS:  35%|██████▋            | 1.27G/3.58G [04:26<08:04, 4.76MB/s] 


The dataset base class and testing class (**for developers**): 

.. autosummary::
    :toctree: autodoc

    EasyidpDataSet
    TestData


Functions
=========

.. autosummary:: 
    :toctree: autodoc

    user_data_dir
    show_data_dir
    url_checker
    download_all


The functions can be used by:

.. code-block:: python

    >>> import easyidp as idp
    >>> idp.data.user_data_dir()
    PosixPath('/Users/<user>/Library/Application Support/easyidp.data')
    >>> idp.data.show_data_dir()
    # please check the popup file explorer