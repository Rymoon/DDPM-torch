# Something not depend on pytorch 
# From RenNet, 23-04-30

import os
from pathlib import Path
import json

'''
From RenNet/env.py,



Example of apps/rennet.json:
{
    "root_Results":"/.../Results",
    "datasets":{
        "CelebAHQ256":{
            "imgs":"/.../Dataset/CelebAHQ/data256x256/",
            "suffix":"jpg"
            },
        "CelebAHQ256_valid":{
            "imgs":"/.../Dataset/CelebAHQ/data256x256_valid/",
            "suffix":"jpg"
        },
        "CelebAHQ256_1":{
            "imgs":"/.../Dataset/CelebAHQ_1/",
            "suffix":"jpg"
        },
        "CelebAHQ256_2":{
            "imgs":"/.../Dataset/CelebAHQ_2/",
            "suffix":"jpg"
        }
    }
}

'''
with open(Path(Path(__file__).parent,f"{Path(__file__).stem}.json").as_posix()) as f:
    _d=  json.loads(f.read())

root_Results=  _d["root_Results"]
datasets = _d["datasets"]

'''
2021-2-5, RenNet/framework/Core/RyCore/__init__
'''
from datetime import datetime
import hashlib
import regex
import inspect
import json
import os
import shutil
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from inspect import currentframe, getframeinfo, signature
from typing import (Generic, Iterable, List, Optional, Tuple, TypeVar, Union,
                    overload)


# from RenNet.framework.Core import console
# print = console.log

def EMPTY_FUNCTION(*args,**kargs):
    pass
ENDL = "\n"
#


def print_notice(*obj_list):
    """
    Green"""
    print(*obj_list,style="green")

def print_error(*obj_list):
    """
    Red"""
    print(*obj_list,style="red")


# PLOT Time

def sec2hr(sec:float):
    """
    h,m,s: float,float,float
    """
    h = sec//60//60
    m = (sec//60)%60
    s = sec%60
    return h,m,s

def getTimestr():
    '''
    2021-02-08T11:56:15.762
    '''
    
    now = datetime.now() #obj_like datetime(2022, 3, 22, 15, 36, 13, 369673)
    now_str = f"{now.year}-{now.month:0>2d}-{now.day:0>2d}-T{now.hour:0>2d}:{now.minute:0>2d}:{now.second:0>2d}.{now.microsecond//1000:0>3d}"
    
    return now_str

def getTimestrAsFilename():
    """
    yyyy-MM-dd-hh-mm-ss-zzz
    """
    now = datetime.now() #obj_like datetime(2022, 3, 22, 15, 36, 13, 369673)
    now_str = f"{now.year}-{now.month:0>2d}-{now.day:0>2d}-{now.hour:0>2d}-{now.minute:0>2d}-{now.second:0>2d}-{now.microsecond//1000:0>3d}"
    
    return now_str

# PLOT File

def getcfp(s:str):
    p = os.path.split(s)[0]
    return p





def is_valid_keys(keys,valid_keys:List[str],check_name:Optional[str]=None,raise_exception=True):

    if check_name is None:
        check_name ="is_valid_keys"
    invalid_keys = []

    for k in keys:
        if k not in valid_keys:
            invalid_keys.append(k)
    if raise_exception:
        assert len(invalid_keys)==0,f"{check_name}: {invalid_keys} not in {valid_keys}."
    else:
        return invalid_keys

def has_keys(required_keys:List[str],target_dict:dict,dict_name:Optional[str]=None,raise_exception =True):
    
    if dict_name is None:
        dict_name = 'has_keys'
    missing_keys = []
    for k in required_keys:
        if k not in target_dict:
            missing_keys.append(k)
    if raise_exception:
        assert len(missing_keys)==0,f"{dict_name}:{missing_keys} not found."
    else:
        return missing_keys


# PLOT Console
from enum import Enum


class Color(Enum):
    red=  'red'
    green = 'green'

def COLOR(s:str,color:Color=Color.red)->str:
    '''
    r   red
    g   green
    '''
    if color == Color.red:
        r= '\033[5;31;40m{}\033[0m'.format(s)
    elif color ==Color.green:
        r = '\033[4;32;40m{}\033[0m'.format(s)
    else:
        r = s
    return r





# PLOT Data structure, data easy-fetch
_T_co= TypeVar('_T_co')
class batch(Generic[_T_co]):
    """Iterator.
    Yield tuple of length sz_batch.
    The last yielded-value may smaller than sz_btach.
    """
    def __init__(self,it:Iterable[_T_co],sz_batch:int):
        self.it = it
        self.sz_batch = sz_batch

    def __iter__(self):
        while True:
            cache =tuple(obj for i,obj 
                in zip(range(self.sz_batch),self.it))
            if len(cache)==0:
                #raise StopIteration()
                return
            yield cache


class LazyList:
    """
    A list of callable things.
    
    [f0,f1,...]

    lazydict[0] --> f0(0)
    
    
    
    
    .. note::
        **When to use?**
    
        We will lose references if store buffer-tensors in built-in list, when
            - Use `model.to(device)` to move all buffer-tensors. 
        
        PyTorch has registered `Parameter` to avoid such reference loss, but no `Buffer` class.
        
        See Parameter and ParamterList
            
        `LazyList,LazyDict` will call getattr immediately when getitem. 
    """
    def __init__(self,value):
        self.data = list(value)
    def __getitem__(self,key):
        f = self.data[key]
        return f(key)
    def __len__(self):
        return len(self.data)
    def __setitem__(self,key,value):
        self.data[key] = value

class LazyDict:
    """
    A dict of Callable things.

    {key:fkey,...}

    lazydict[key] --> fkey(key)
    
    .. note::
        
        **When to use?**
    
        See class `LazyList`
    """

    def __init__(self,value):
        self.data = dict(value)
    def __getitem__(self,key):
        f = self.data[key]
        return f(key)
    def __len__(self):
        return len(self.data)
    def __setitem__(self,key,value):
        self.data[key] = value



import collections
from typing import Callable,Iterable

import Levenshtein
def find_nearest_altkey(key,known_keys:Iterable):
    """
    Return altkey, dist

    dist ==0 , if key == altkey
    dist ==-1, if known_keys is empty
    """
    if key in known_keys:
        return key,0
    else:
        distl = []
        kl=[]
        i=-1
        for i,k in enumerate(known_keys):
            d = Levenshtein.distance(k,key)
            distl.append(d)
            kl.append(k)
        if i==-1:
            return None,-1 # known_keys is empty
        else:
            min_d = min(distl)
            j= distl.index(min_d)
            altkey = kl[j]
    return altkey,min_d

class MsgedDict(collections.UserDict):
    def __init__(self,*args,**kargs):
        super().__init__(*args,**kargs)
        self.msg_missingkey = None
        self.handler_missingkey = None
    def set_msg_missingkey(self,msg:str):
        """
        msg: string_template(key=key)
        """
        self.msg_missingkey = msg
    
    def set_handler_missingkey(self,f:Callable):
        """
        f(key:str,known_keys:Iterable)
        """
        self.handler_missingkey = f


    def __getitem__(self, key:str):
        m = self.msg_missingkey
        f = self.handler_missingkey
        if key not in self:
            if f is not None:
                msg=f(self,key)
            else:
                altkey,dist = find_nearest_altkey(key,self.keys())
                if dist ==-1:
                    msg = f"Key '{key}' not found. It's an empty dict."
                else:
                    msg = f"Key '{key}' not found. Do you mean '{altkey}' ?"
            raise KeyError(msg)
        else:
            return super().__getitem__(key)


def kwargs(*keywords):
    """
    NOTICE: Using @kwargs() 
    
    For dataclass to have keyword-arguments only restriction(Before Python 3.10)

    if len(keywords)==0 then ALL_KEYWORDS_ONLY

    https://stackoverflow.com/questions/49908182/how-to-make-keyword-only-fields-with-dataclasses

    Coming in Python 3.10, there's a new dataclasses.KW_ONLY sentinel that works like this:
    
    .. code-block:: python
    
        @dataclasses.dataclass
        class Example:
            a: int
            b: int
            _: dataclasses.KW_ONLY
            c: int
            d: int
    
    Any fields after the KW_ONLY pseudo-field are keyword-only.

    There's also a kw_only parameter to the dataclasses.dataclass decorator, which makes all fields keyword-only:

    .. code-block:: python
    
        @dataclasses.dataclass(kw_only=True)
        class Example:
            a: int
            b: int
    
    It's also possible to pass kw_only=True to dataclasses.field to mark individual fields as keyword-only.

    If keyword-only fields come after non-keyword-only fields (possible with inheritance, or by individually marking fields keyword-only), keyword-only fields will be reordered after other fields, specifically for the purpose of __init__. Other dataclass functionality will keep the declared order. This reordering is confusing and should probably be avoided."""
    
    def decorator(cls):
        @wraps(cls)
        def call(*args, **kwargs):
            sig = signature(cls)
            param_l = list(sig.parameters.keys())

            if len(keywords) == 0:
                kw_l = param_l
            elif any(kw not in param_l for kw in keywords):
                raise Exception(f"Decorator: Not all {keywords} in {cls.__name__}({param_l}).")
            else:
                kw_l = keywords

            for kw_in_need in kw_l:
                if kw_in_need not in kwargs and sig.parameters[kw_in_need].default != inspect.Signature.empty:

                    raise TypeError(f"{cls.__name__}.__init__() requires {kw_in_need} as keyword arguments")
            
            n_pos_or_kw = len(param_l)-len(kw_l)
            if len(args)>n_pos_or_kw:
                raise TypeError(f"{cls.__name__}.__init__() requires {kw_l} as keyword arguments")
            return cls(*args, **kwargs)
        
        return call

    return decorator

import re

def splitkeys(keystr:str,*,delim:str=" "):
    """
    space at beg/end will be ignored;
    delim at beg/end will be ignored;
    newline will be ignored;
    double-delim will be ignored;
    """
    try:
        ks = keystr
        assert len(delim) ==1
        ks = ks.replace("\n",delim)
        ks = re.sub(f"{delim}[ ]{{0,}}",f"{delim}",ks)
        ks = re.sub(f"[ ]{{0,}}{delim}",f"{delim}",ks)
        ks = re.sub(f"[{delim}]{{2,}}",f"{delim}",ks)
        ks = ks.lstrip().rstrip()
        ks = ks.lstrip(delim).rstrip(delim)
        keys = ks.split(delim)
    except Exception as e:
        print(f"== keystr = {keystr}")
        print(f"== `delim` = `{delim}`")
        import traceback
        print(traceback.format_exc())
        raise e
    return keys 


def _getitems(dictlike,keystr:str,*,delim:str=" "):
    """
    Prototype:
    - return tuple(keys), tuple(values)
    - `s[0]` or `s[text]` will turn to `s.0` or `s.text` as keys 

    You should use one of these notations in keystr, not a mixture:
    - "x y z a"
    - "x[0] y[0] z[0]"
    - "x[alpha] k[beta]"

    No check on this! Undefined behaviour.

    If you musr mix notations, re-factor your data structure! A sign of bad codes :-)


    dictlike:
    - dict
    - vars(dataclass)
    - dict(module.named_buffers())
    
    See `splitkeys(keystr,delim)`
    
    If '[' in the first `k`, will catch a k2, and
    - take d[k][int(k2)], then
    - if failed, will take d[k][str(k2)].

    Notice that the k2-logic(int or str) MUST be the same for all keys.

    For dataclasses, use `var(dataclass_obj)` as `dictlike`
 

    """
    keys = splitkeys(keystr,delim=delim)
    assert len(keys)>=1,f"{keys}"
    if "[" in keys[0]:
        # check k2-logic
        k,k2 = keys[0].split("[")
        k2= k2[:-1]
        try:
            int(k2)
            isint=True
        except ValueError as e:
            str(k2)
            isint=False

        kp = tuple(key.split("[") for key in keys)
        kp = tuple( (k,k2[:-1]) for k,k2 in kp)

        
        
        try:
            if isint:
                tp= tuple(dictlike[k][int(k2)] for k,k2 in kp)
            else:
                tp= tuple(dictlike[k][str(k2)] for k,k2 in kp)
        except Exception as e:
            print(f"== kp={kp}")
            import traceback
            print(traceback.format_exc())
            raise e

        return tuple(f"{k}.{k2}" for (k,k2) in kp), tp
    else:
        return keys, tuple(dictlike[k] for k in keys)

from typing import Dict
def getitems(dictlike,keystr:str,*,delim:str=" ")->tuple:
    """
    dictlike:
    - dict
    - vars(dataclass)
    - dict(module.named_buffers())

     You should use one of these notations in keystr, not a mixture:
    - "x y z a"
    - "x[0] y[0] z[0]"
    - "x[alpha] k[beta]"

    No check on this! Undefined behaviour.
    

    Example:

    ```python
    getitems({"a":0,"b":1},"  a b ") # (0,1)

    getitems(var(object),"names[0], tags[0]",delim=",") # (obj.names[0], obj.tags[0])
    ```

    See `splitkeys(keystr,delim)` and `_getitems` for more.
    

    """
    return _getitems(dictlike,keystr,delim=delim)[1]


def getitems_as_dict(dictlike,keystr:str,*,delim:str=" "):
    """
    dictlike:
    - dict
    - vars(dataclass)
    - dict(module.named_buffers())

     You should use one of these notations in keystr, not a mixture:
    - "x y z a"
    - "x[0] y[0] z[0]"
    - "x[alpha] k[beta]"

    No check on this! Undefined behaviour.
    

    Example:

    ```python
    getitems({"a":0,"b":1},"  a b ") # {"a":0,"b":1}

    getitems(var(object),"names[0], tags[0]",delim=",") # ("names.0": obj.names[0], "tags.0": obj.tags[0])
    ```

    See `splitkeys(keystr,delim)` and `_getitems` for more.

    """
    try:
        kl,vl = _getitems(dictlike,keystr,delim=delim)
    except KeyError as e:
        key = e.args[0]
        print(f"- Error: the key `{key}` not found in dictlike.")
        raise e
    return {k:v for k,v in zip(kl,vl)}

def updateattrs(obj,newdict):
    for k,v in newdict.items():
        setattr(obj,k,v)

def dargs_for_calling(f:Callable,d:dict):
    """Inspect the arg list of Callable f, fetch items for d and return;
    """
    import inspect
    args = inspect.getfullargspec(f)
    fal = [n for n in args.args]+[n for n in args.kwonlyargs]
    fd=  {}
    for k in fal:
        if k in d:
            fd[k] = d[k]
    
    return fd

def call_by_inspect(f:Callable,d:dict,**kwargs):
    """
    dargs = dargs_for_calling(f,d))

    dargs.update(kwargs)

    Return f(**d)
    """
    dargs = dargs_for_calling(f,d)
    dargs.update(kwargs)
    return f(**dargs)

def as_int(s):
    """
    Return None if failed;
    Or, return integer;

    Notice 0(int) is false-if-condition;
    """
    try:
        i = int(s)
    except:
        return None
    else:
        return i
    


def dargs_for_formatting(string_template:str,d:dict):
    """
    Return dict

    ONLY support keyword-format: {a};
    Raise exception if {}, {{}}, {0}
    """
    p = r"({(?:[^{}]*|(?R))*})" # will capture the brackets and contents between;
    s=string_template

    pt = regex.compile(p)
    m = regex.findall(pt,s)
    args = []
    for t in m:
        t2=  t[1:-1]
        if "{" in t2:
            raise Exception(f"Not supported: recursive: {t}")
        if as_int(t2) is not None:
            raise Exception(f"Not supported: positional: {t}")
        args.append(t2)
    dargs = {n:d[n] for n in args if n in d}
    return dargs

def format_by_re(string_template:str,d:dict,**kwargs):
    """Only kwargs-string-format supported.
    """

    dargs = dargs_for_formatting(string_template,d)
    dargs.update(kwargs)
    s2=  string_template.format(**dargs)
    return s2

import json
from pathlib import PosixPath
class RenNetJSONEncoder(json.JSONEncoder):
    """
    - MsgedDict: Dict
    - PosixPath: str
    """
    def default(self, obj):
        if isinstance(obj, MsgedDict):
            return dict(obj)
        elif isinstance(obj,PosixPath):
            return obj.as_posix()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)






