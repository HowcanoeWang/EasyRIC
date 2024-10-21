import json
import os
import numpy as np
import geojson
import pyproj
import warnings
from tabulate import tabulate
from tqdm import tqdm

import easyidp as idp

class MyEncoder(json.JSONEncoder):
    # The original json package doesn't compatible to numpy object, add this compatible encoder to it.
    # usage: json.dump(..., cls=MyEncoder)
    # references: https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def read_json(json_path):
    """Read json file to python dict.

    Parameters
    ----------
    json_path : str
        The path to json file

    Returns
    -------
    dict

    Example
    -------

    Data prepare:

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

    Use this function:

    .. code-block:: python

        >>> out = idp.jsonfile.read_json(test_data.json.for_read_json)
        >>> out
        {'test': {'rua': [[12, 34], [45, 56]]}, 'hua': [34, 34.567]}

    """
    if os.path.exists(json_path):
        with open(json_path) as json_file:
            data = json.load(json_file)
            return data
    else:
        raise FileNotFoundError(f"Could not locate the given json file [{json_path}]")
    
def _check_geojson_format(geojson_path, encoding='utf-8'):
    if '.geojson' not in str(geojson_path):
        raise TypeError(f"The input file format should be *.geojson, not [.{str(geojson_path).split('.')[-1]}]")

    try:
        with open(geojson_path, 'r', encoding=encoding) as f:
            geojson_data = geojson.load(f)
    except UnicodeDecodeError:
        encoding = _check_file_encoding(geojson_path)
        with open(geojson_path, 'r', encoding=encoding) as f:
            geojson_data = geojson.load(f)

    # Only the FeatureCollection geojson type is supported.
    if geojson_data.type != 'FeatureCollection':
        raise TypeError(f'A `FeatureCollection` type geojson is expected, current type is `{geojson_data.type}`')
    
    if 'features' not in geojson_data.keys():
        raise TypeError('geojson does not have features properties')
    
    if len(geojson_data.features) == 0:
        raise IndexError('geojson must have at least 1 item')
    
    return geojson_data

def _check_file_encoding(file):
    import chardet
    # check file encode
    with open(file, 'rb') as f:
        encode_result = chardet.detect(f.read())
        print(f'Detect input file [{file}] encoding is "{encode_result}".')
        return encode_result
    
def read_geojson(geojson_path, name_field=-1, include_title=False, return_proj=False):
    """Read geojson file to python dict

    Parameters
    ----------
    geojson_path : str
        The path to geojson file
    name_field : str or int or list[ str|int ], optional
        by default None, the id or name of shp file fields as output dictionary keys
    include_title : bool, optional
        by default False, whether add column name to roi key.
    return_proj : bool, optional
        by default False, if given as true, will return extra pyproj.CRS object of current shp file.

    Returns
    -------
    dict, 
        the dictionary with read numpy polygon coordinates

        .. code-block:: python

            {'id1': np.array([[x1,y1],[x2,y2],...]),
             'id2': np.array([[x1,y1],[x2,y2],...]),...}
    pyproj.CRS, optional
        once set return_proj=True
    
    Example
    -------

    The example geojons file has the following columns:

    .. code-block:: text

         [0] FID    [1] 試験区    [2] ID    [3] 除草剤    [4] plotName    [5] lineNum
        ---------  ------------  --------  ------------  --------------  -------------
           65       SubBlk 2b       0           有          Enrei-10           1
           97       SubBlk 2b       0           有          Enrei-20           1
           147      SubBlk 2b       0           有        Nakasenri-10         1
           ...         ...         ...         ...            ...             ...
           259        SB 0a         0                    Tachinagaha-10        3
            4         SB 0a         0                    Fukuyutaka-10         3
            1       SubBlk 2a       0           無          Enrei-20           1

    Data prepare:

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()
        >>> data_path = test_data.json.geojson_soy

    Use this function to read geojson:

    .. code-block:: python

        >>> out = idp.jsonfile.read_geojson(data_path)
        >>> out['196']  
        array([[-26403.247163, -28847.161464],
               [-26404.137868, -28843.261904],
               [-26404.820258, -28843.417749],
               [-26403.929552, -28847.317309],
               [-26403.247163, -28847.161464]])

    .. caution::

        The ``196`` in previous code is the row 196, not the ``FID``=196. 
        To use the FID as result key, Please use ``name_field`` introduced below
               
    If want use other attributes as result keys, for example, the first column ``FID`` as the key:

    .. code-block:: python

        >>> out = idp.jsonfile.read_geojson(data_path, name_field="FID")
        >>> # or
        >>> out = idp.jsonfile.read_geojson(data_path, name_field="0")
        >>> out['196']
        array([[-26391.733164, -28862.99536 ],
               [-26392.623851, -28859.09578 ],
               [-26393.306263, -28859.25163 ],
               [-26392.415575, -28863.15121 ],
               [-26391.733164, -28862.99536 ]])

    You can also combine multiple columns as the key values, by passing a list to ``name_field``:

    .. code-block:: python

        >>> out = idp.jsonfile.read_geojson(data_path, name_field=["FID", "plotName"])
        >>> # or
        >>> out = idp.jsonfile.read_geojson(data_path, name_field=[0, 4])
        >>> out.keys()
        dict_keys(['65_Enrei-10', '97_Enrei-20', '147_Nakasenri-10', ... ])

    And you can also add column_names to id by ``include_title=True`` :

    .. code-block:: python

        >>> out = idp.jsonfile.read_geojson(data_path, name_field=["FID", "plotName"], include_title=True)
        >>> # or
        >>> out = idp.jsonfile.read_geojson(data_path, name_field=[0, 4], include_title=True)
        >>> out.keys()
         dict_keys(['FID_65_plotName_Enrei-10', 'FID_97_plotName_Enrei-20', ... ])

    See also
    --------
    easyidp.jsonfile.show_geojson_fields, easyidp.shp.read_shp

    """
    geo_dict = {}
    crs_proj = None

    geojson_data = _check_geojson_format(geojson_path)
    
    if 'crs' in geojson_data.keys():
        # ensure it has correct format for CRS info
        if  'properties' in geojson_data.crs and \
            'name'       in geojson_data.crs['properties']:
            crs_proj = pyproj.CRS.from_string(geojson_data.crs['properties']['name'])
        else:
            crs_template = {
                "type": "name",
                "properties": {
                    "name": "EPSG:xxxx"
                }
            }
            print(f'geojson does not have standard CRS properties like:\n{crs_template}'
                  f'\nBut the following is obtained:\n{geojson_data.crs}')
    else:
        print(f'[json][geojson] geojson does not have CRS properties')

    # calculate the geo_fields (shp_fields)
    # Format:  {"Column": int_id}
    # Exmaple: {"ID":0, "MASSIFID":1, "CROPTYPE":2, ...}
    geo_fields = {}
    for i, key in enumerate(geojson_data.features[0]['properties'].keys()):
        geo_fields[key] = i

    ### do not put it into the following loop, save calculation time.
    # field_id => int or list[ int ] 
    if isinstance(name_field, list):
        field_id = [idp.shp._find_name_related_int_id(geo_fields, nf) for nf in name_field]
    else:
        field_id = idp.shp._find_name_related_int_id(geo_fields, name_field)  

    # build the format template
    plot_name_template, keyring = idp.shp._get_plot_name_template(geo_fields, field_id, include_title)

    non_polygon_warning = 0
    pbar = tqdm(
        geojson_data.features,
        desc=f"[json] Read geojson [{os.path.basename(geojson_path)}]"
    )
    for i, feature in enumerate(pbar):
        # convert dict_key name string by given name_field

        ### feature['property'] => 
        # {'FID': 65,
        #  '試験区': 'SubBlk 2b',
        #  ...
        #  'lineNum': 1}

        if isinstance(field_id, list):
            # list comprehension
            values = [
                feature['properties'][_key] 
                if fid != -1 else i 
                for fid, _key in zip(field_id, keyring)
            ]
            
            plot_name = plot_name_template.format(*values)
        else:
            if field_id != -1:
                plot_name = plot_name_template.format(feature['properties'][keyring])
            else:
                plot_name = plot_name_template.format(i)

        plot_name = plot_name.replace(r'/', '_')
        plot_name = plot_name.replace(r'\\', '_')

        ##################################
        # get the shape coordinate value #
        ##################################
        geometry = feature['geometry']
        # -> 
        # {"coordinates": [[[-26384.952573, -28870.678514], 
        #                   [-26384.269447, -28870.522501], 
        #                   [-26385.160022, -28866.622912], 
        #                   [-26385.843163, -28866.778928], 
        #                   [-26384.952573, -28870.678514]]], 
        #  "type": "Polygon"}

        if geometry['type'] == "Polygon":
            # Polygon -> nx2x1 lists, only need nx2
            coord_np = np.asarray(geometry['coordinates'][0])
        else:
            non_polygon_warning += 1
            # only warning at the first time
            if non_polygon_warning == 1:
                warnings.warn(f"Currently only supports [Polygon] type geojson, but [{geometry['type']}] are used")
            
            coord_np = np.asarray(geometry['coordinates'])

        # check if has duplicated key, otherwise will cause override
        if plot_name in geo_dict.keys():
            raise KeyError(f"Meet with duplicated key [{plot_name}] for current shapefile, please specify another `name_field` from {geo_fields} or using row id as key `name_field='#'`")

        geo_dict[plot_name] = coord_np

    if return_proj:
        return geo_dict, crs_proj
    else:
        return geo_dict


def show_geojson_fields(geojson_path):
    """
    Show geojson properties data, for better setting ``name_field`` of :py:obj:`read_geojson <easyidp.roi.ROI.read_geojson>`

    Parameters
    ----------
    geojson_path : : str
        the file path of \*.geojson

    Example
    -------

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> idp.jsonfile.show_geojson_fields(test_data.json.geojson_soy)
         [0] FID    [1] 試験区    [2] ID    [3] 除草剤    [4] plotName    [5] lineNum
        ---------  ------------  --------  ------------  --------------  -------------
           65       SubBlk 2b       0           有          Enrei-10           1
           97       SubBlk 2b       0           有          Enrei-20           1
           147      SubBlk 2b       0           有        Nakasenri-10         1
           ...         ...         ...         ...            ...             ...
           259        SB 0a         0                    Tachinagaha-10        3
            4         SB 0a         0                    Fukuyutaka-10         3
            1       SubBlk 2a       0           無          Enrei-20           1
    
    See also
    --------
    easyidp.shp.show_shp_fields

    """
    geojson_data = _check_geojson_format(geojson_path)

    head = ["[-1] #"] + \
        [f"[{i}] {k}" for i, k in enumerate(geojson_data.features[0]['properties'].keys())]
    data = []

    row_num = len(geojson_data.features)
    col_num = len(geojson_data.features[0]['properties'].keys())

    col_align = ["right"] + ["center"] * col_num

    if row_num >= 6:
        show_idx = [0, 1, 2, -3, -2, -1]
    else:
        # print all without omit
        show_idx = list(range(row_num))

    for i in show_idx:
        if i >= 0:
            data.append(
                [i] + list(geojson_data.features[i]['properties'].values())
            )
        else:
            data.append(
                [row_num + i] + list(geojson_data.features[i]['properties'].values())
            )

    if row_num > 6:
        data.insert(3, ['...'] * (col_num + 1))

    table_str = tabulate(data, headers=head, tablefmt='simple', colalign=col_align)
    print(table_str)


def dict2json(data_dict, json_path, indent=None, encoding='utf-8'):
    """Save dict object to the same structure json file
    
    Parameters
    ----------
    data_dict : dict
        the dict object want to save as json file
    json_path : str
        the path including json file name to save the json file
        e.g. ``D:/xxx/xxxx/save.json`` 
    indent : int | None
        whether save "readable" json with indent, default 0 without indent
    encoding : str
        the encoding type of output file

    Example
    -------

    .. code-block:: python

        >>> import easyidp as idp
        >>> a = {"test": {"rua": np.asarray([[12, 34], [45, 56]])}, "hua":[np.int32(34), np.float64(34.567)]}
        >>> idp.jsonfile.dict2json(a, "/path/to/save/json_file.json")

    .. note:: 

        Dict without indient:

        .. code-block:: python

            >>> print(json.dumps(data), indent=0)
            {"age": 4, "name": "niuniuche", "attribute": "toy"}

        Dict with 4 space as indient:

        .. code-block:: python

            >>> print(json.dumps(data,indent=4))
            {
                "age": 4,
                "name": "niuniuche",
                "attribute": "toy"
            }

    See also
    --------
    easyidp.jsonfile.write_json, easyidp.jsonfile.save_json
    
    """
    json_path = str(json_path)
    if isinstance(json_path, str) and json_path[-5:] == '.json':
        with open(json_path, 'w', encoding=encoding) as result_file:
            json.dump(data_dict, result_file, ensure_ascii=False, cls=MyEncoder, indent=indent)

            # print(f'Save Json file -> {os.path.abspath(json_path)}')


def write_json(data_dict, json_path, indent=0, encoding='utf-8'):
    """Save dict to the same structure json file, a function wrapper for :func:`dict2json`
    
    Parameters
    ----------
    data_dict : dict
        the dict object want to save as json file
    json_path : str
        the path including json file name to save the json file
        e.g. ``D:/xxx/xxxx/save.json``
    indent : int | None
        whether save "readable" json with indent, default 0 without indent
    encoding : str
        the encoding type of output file


    See also
    --------
    easyidp.jsonfile.dict2json

    """
    dict2json(data_dict, json_path, indent, encoding)


def save_json(data_dict, json_path, indent=0, encoding='utf-8'):
    """Save dict to the same structure json file, a function wrapper for :func:`dict2json`
    
    Parameters
    ----------
    data_dict : dict
        the dict object want to save as json file
    json_path : str
        the path including json file name to save the json file
        e.g. ``D:/xxx/xxxx/save.json``
    indent : int | None
        whether save "readable" json with indent, default 0 without indent
    encoding : str
        the encoding type of output file

    See also
    --------
    easyidp.jsonfile.dict2json

    """
    dict2json(data_dict, json_path, indent, encoding)

# just copied from previous `caas_lite.py`, haven't modified yet
def _to_labelme_json(grid_tagged, json_folder, minimize=True):
    """Save the tagged shp polygon crop result to json file, for deeplearing use

    Parameters
    ----------
    grid_tagged : pandas.DataFrame
        the output of self.dataframe_add_shp_tags()
        The 4 column dataframe shows in this function introduction, sorted by "grid_name"
    json_folder : str
        the folder or path to save those json files
    minimize : bool
        whether create a json without space

    Notes
    -----
    The labelme json file has the following structure:

    .. code-block:: json

        {
            "version": "4.5.6",  # the Labelme.exe version, optional
            "flags": {},
            "imagePath": "xxxx.tiff",
            "imageHeight": 1000,
            "imageWidth": 1000,
            "imageData": null,
            "shapes": [{ }, { }, { }]
        }      

    for each ``{}`` items in "shapes":

    .. code-block:: json

        {
            "label": "field",
            "group_id": null,
            "shape_type": "polygon",
            "flags": {},
            "points": [[x1, y1], [x2, y2], [x3, y3]]  # with or without the first point
        }

    Example of ``grid_tagged`` 

    +------------------+----------+---------------------+-------+
    |     grid_name    | dict_key |    polygon_list     |  tag  |
    +==================+==========+=====================+=======+
    | 'grid_x1_y1.tif' | 'key1'   | [poly1, poly2, ...] | field |
    +------------------+----------+---------------------+-------+
    | 'grid_x1_y1.tif' | 'key2'   | [poly1, poly2, ...] | crops |
    +------------------+----------+---------------------+-------+
    | 'grid_x1_y2.tif' | 'key1'   | [poly1, poly2, ...] | field |
    +------------------+----------+---------------------+-------+
    | 'grid_x2_y1.tif' | 'key1'   | [poly1, poly2, ...] | field |
    +------------------+----------+---------------------+-------+
    | 'grid_x3_y2.tif' | 'key2'   | [poly1, poly2, ...] | crops |
    +------------------+----------+---------------------+-------+
    |      ...         |   ...    |         ...         |  ...  | 
    +------------------+----------+---------------------+-------+

    Example of ``minimize``

    - True

      .. code-block:: json

        {"name":"lucy","sex":"boy"}
    
    - False

      .. code-block:: json

        {
            "name":"lucy",
            "sex":"boy"
        } 

    """
    total_dict = {}
    for i in range(len(grid_tagged)):
        # todo modify here
        img_name = grid_tagged.loc[i]['grid_name']
        poly = grid_tagged.loc[i]['polygon_list']
        tag = grid_tagged.loc[i]['tag']
        
        if img_name not in total_dict.keys():
            single_dict = {"version": "4.5.6", 
                            "flags": {},
                            "imagePath": img_name,
                            "imageHeight": 1000,
                            "imageWidth": 1000,
                            "imageData": None,
                            "shapes": []}
        else:
            single_dict = total_dict[img_name]
            
        for item in poly:
            single_item = {"label": tag,
                            "group_id": None,   # json null = python None
                            "shape_type": "polygon",
                            "flags": {},
                            "points": item.tolist()}
            single_dict['shapes'].append(single_item)
            
        total_dict[img_name] = single_dict
        
    # after iter all items
    for k, d in total_dict.items():
        json_name = k.replace('.tif', '.json')
        if minimize:
            dict2json(d, os.path.join(json_folder, json_name))
        else:
            dict2json(d, os.path.join(json_folder, json_name), indent=2)