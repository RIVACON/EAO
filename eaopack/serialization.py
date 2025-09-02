import numpy as np
import pandas as pd
import datetime as dt
import json
import pytz

from eaopack.assets import *     # need to import to be able to deserialize
from eaopack.portfolio import *  # need to import to be able to deserialize
from eaopack.io import extract_output, output_to_file


def json_serialize_objects(obj) -> dict:
    """ serialization function for JSON
    Args:
        obj ([type]): object to be serialized
    Returns:
        dict: serialized object for json
    """
    # simple conversion for some types
    # hook
    if isinstance(obj, dt.datetime) or isinstance(obj, pd.Timestamp):
        if obj.tzinfo is None:
            mytz = None
        else:
            # saving as UTC
            mytz = str(obj.tzinfo)
            obj = pd.Timestamp(obj).tz_convert('UTC')
        res =  {'__class__': dt.datetime.__name__,
                '__tz__'   : mytz,
                '__value__' : obj.strftime("%Y-%m-%d %H:%M:%S")
               }
    elif isinstance(obj, dt.date):
        res =  {'__class__': dt.date.__name__,
                '__value__': str(obj)
               }               
    elif isinstance(obj, Unit):
        res = obj.__dict__.copy()
        res['__class__'] = 'Unit'
    elif isinstance(obj, Node):
        res = obj.__dict__.copy()
        res['__class__'] = 'Node'
    elif isinstance(obj, Timegrid):
        res = {'__class__' : 'Timegrid',
               'start'     : obj.__dict__['start'],
               'end'       : obj.__dict__['end'],               
               'freq'      : obj.__dict__['freq'],               
               'main_time_unit'     : obj.__dict__['main_time_unit']
               }
    elif isinstance(obj, Asset):
        res = obj.__dict__.copy()
        res.pop('timegrid',None)
        res.pop('asset_names',None)
        res.pop('start_ramp_time', None)  # only relevant for class CHPAsset, calculated
        res.pop('shutdown_ramp_time', None)  # only relevant for class CHPAsset, calculated
        res.pop('idx_nodes', None)  # only relevant for class CHPAsset, calculated        
        # res.pop('timegrid', None) # not to be serialized
        res['__class__']  = 'Asset' # super class Asset
        res['asset_type'] = obj.__class__.__name__ # store child class
        if res['asset_type'] == 'OrderBook': # some parameters not relevant
            res.pop('start', None)
            res.pop('end', None)
            res.pop('freq', None)
            res.pop('profile', None)
    elif isinstance(obj, Portfolio):
        res = {'assets': obj.assets}
        if hasattr(obj, 'timegrid'):
            res['timegrid'] = obj.timegrid
        res['__class__']  = 'Portfolio' # super class Asset
    elif isinstance(obj, np.ndarray):
        res = {'__class__' : 'np_array'}
        res['is_date'] = np.issubdtype(obj.dtype, np.datetime64)
        # Note: For datetime this leads to saving a number in JSON. May want to make it a str
        res['np_list'] =  obj.tolist() 
    elif isinstance(obj, pd.DatetimeIndex):
        res = {'__class__' : 'pd_DateTimeIndex',
               '__freq__'  : obj.freqstr,
               '__value__' : obj.tolist()}
    else:
        raise TypeError(str(obj) + ' is not json serializable')
    return res

def json_deserialize_objects(obj):   
    if '__class__' in obj:
        if obj['__class__'] == 'datetime':
            res = pd.Timestamp(dt.datetime.strptime(obj['__value__'], "%Y-%m-%d %H:%M:%S"))
            if '__tz__' in obj:
                if not obj['__tz__'] is None:
                    # saved as UTC
                    res = res.tz_localize('UTC')
                    res = res.tz_convert(obj['__tz__'])
        elif obj['__class__'] == 'date':
            res = dt.datetime.strptime(obj['__value__'], "%Y-%m-%d").date()
        elif obj['__class__'] == 'Node':
            obj.pop('__class__', None)
            res = Node(**obj)
        elif obj['__class__'] == 'Unit':
            obj.pop('__class__', None)
            res = Unit(**obj)
        elif obj['__class__'] == 'Timegrid':
            obj.pop('__class__', None)            
            res = Timegrid(**obj)                        
        elif obj['__class__'] == 'Asset':
            obj.pop('__class__', None)
            obj.pop('timegrid', None)
            asset_type = obj['asset_type']
            obj.pop('asset_type', None)
            res = globals()[asset_type](**obj)
        elif obj['__class__'] == 'Portfolio':
            obj.pop('__class__', None)
            res = Portfolio(obj['assets'])
            if 'timegrid' in obj:
                res.set_timegrid(obj['timegrid'])
        elif obj['__class__'] == 'np_array':
            if 'is_date' in obj: # backwards compatible
                if obj['is_date']:
                    obj['np_list'] = [np.datetime64(ll, 'ns') for ll in obj['np_list']]
            res = np.asarray(obj['np_list'])
        elif obj['__class__'] == 'pd_DateTimeIndex':
            res = pd.DatetimeIndex(obj['__value__'], freq = obj['__freq__'])
        else:
            raise NotImplementedError(obj['__class__']+ ' not deseralizable')
    else:
        res = obj
    return res

def to_json(obj, file_name = None):
    """ serialize object to JSON and save to file

    Args:
        obj: json serializable object
        file_name (str): Filename. Defaults to None (return str)
    """
    if file_name is None:
        return json.dumps(obj, indent=4, sort_keys=True,default=json_serialize_objects)
    else:
        with open(file_name, "w") as file:
            json.dump(obj, file, indent=4, sort_keys=True,default=json_serialize_objects)

def load_from_json(json_str:str = None, file_name:str = None):
    """ create object from JSON in file or string
    Args:
        file_name (str): Filename containing json string. Optional
        json_str (str) : json string. Optional
           one of the two must be given
    Returns:
        object
    """
    if not file_name is None:
        with open(file_name, "r") as file:
            return json.load(file, object_hook=json_deserialize_objects)
    elif not json_str is None:
        return json.loads(json_str, object_hook=json_deserialize_objects)
    else:
        raise ValueError('Either filename or json string must be given')

def run_from_json(json_str:str = None, file_name_in:str = None, \
                  prices: dict = None, timegrid: Timegrid = None,  \
                  file_name_out:str = None, format_out:str = 'xlsx',\
                  csv_ger:bool = False):
    """ (1) create object from JSON in file or string
        (2) run optimization
        (3) write output to file (if file name given)
    Args:
        file_name_in (str)  : Filename containing json string. Optional
        json_str (str)      : json string. Optional
           one of the two must be given
        prices (dict)       : dict of prices to be used for optimization
        timegrid (Timegrid) : timegrid to be used for optimization. 
                              Defaults to None (if portfolio comes with timegrid)
        file_name_out (str) : file name for output
        format_out (str)    : xlsx, csv. format of output file. Defaults to 'xlsx'
        csv_ger (bool)      : English (False) or German (True) csv format. Defaults to False.
    Returns:
        file_name_out given:    Optimization run successfully (bool)
        no file_name_out given: Results dict
    """
    # (1) create object
    portf = load_from_json(json_str, file_name_in)
    if not isinstance(portf, Portfolio):
        raise ValueError('File does not contain an eao Portfolio')

    # (2) run optimization
    if not timegrid is None:
        portf.set_timegrid(timegrid)
    op    = portf.setup_optim_problem(prices)
    res   = op.optimize()
    if isinstance(res, str):
        print('Not successful. No output written')
    else:
        # (3) extract and write output
        out = extract_output(portf, op, res, prices)
        if not file_name_out is None:
            output_to_file(out, file_name_out, format_out,csv_ger)
            return not isinstance(res, str) # opt successful?
        else:
            # return dict with results
            return out
