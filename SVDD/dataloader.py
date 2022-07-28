import h5py
import numpy as np
from tqdm import tqdm    

import sys

def unpack(fname, Flags,precision='float32'):

    files = []

    files.append(fname)

    it = 0
    for filename in tqdm(files, leave=False):
                        
       print ('Unpacking file', filename) 
    
       hf = h5py.File(filename, 'r')

       etype_it = hf.get('etype')
       nobj_it = hf.get('nobj')
       reg_it = hf.get('reg')
       clas_it = hf.get('clas')
   
       etype_it = np.array(etype_it)
       nobj_it = np.array(nobj_it)
       reg_it = np.array(reg_it)
       clas_it = np.array(clas_it)

       if nobj_it.shape[0] == 0 or reg_it.shape[0] == 0 or clas_it.shape[0] == 0:
        continue

       reg_it = reg_it.astype(precision)
       etype_it.astype(precision)


       if it == 0:
        etype = etype_it
        nobj = nobj_it
        reg = reg_it
        clas = clas_it
       else:
        etype = np.concatenate((etype, etype_it))
        nobj = np.concatenate((nobj, nobj_it))
        reg = np.concatenate((reg, reg_it))
        clas = np.concatenate((clas, clas_it))

       if Flags.categorical == 'True':
         clas[clas > 0] = 1

       print ('event type', etype.dtype, etype.shape, float(etype.nbytes)/1.e6, 'MB')   
       print ('numb objects', nobj.dtype, nobj.shape, float(nobj.nbytes)/1.e6, 'MB')
       print ('regression', reg.dtype, reg.shape, float(reg.nbytes)/1.e6, 'MB')
       print ('classification', clas.dtype, clas.shape, float(clas.nbytes)/1.e6, 'MB')

    return etype, nobj, reg, clas

def unpack_ordered(fname, Flags,precision='float32'):

    files = []

    files.append(fname)

    it = 0
    for filename in tqdm(files, leave=False):
                        
       print ('Unpacking file', filename) 
    
       hf = h5py.File(filename, 'r')

       etype_it = hf.get('etype')
       reg_it = hf.get('reg')
   
       etype_it = np.array(etype_it)
       reg_it = np.array(reg_it)

       if reg_it.shape[0] == 0:
        continue

       reg_it = reg_it.astype(precision)

       if it == 0:
        etype = etype_it
        reg = reg_it
       else:
        etype = np.concatenate((etype, etype_it))
        reg = np.concatenate((reg, reg_it))
       
       
       etype.astype(precision)
       reg.astype(precision)

       print ('event type', etype.dtype, etype.shape, float(etype.nbytes)/1.e6, 'MB')   

       print ('regression', reg.dtype, reg.shape, float(reg.nbytes)/1.e6, 'MB')

    return etype, reg

def unpack_polina(fname, Flags,precision='float32'):

    files = []

    files.append(fname)

    it = 0
    for filename in tqdm(files, leave=False):
                        
       print ('Unpacking file', filename) 
    
       hf = h5py.File(filename, 'r')

       etype_it = hf.get('etype')
       nobj_it = hf.get('nobj')
       reg_it = hf.get('reg')
   
       etype_it = np.array(etype_it)
       nobj_it = np.array(nobj_it)
       reg_it = np.array(reg_it)

       if nobj_it.shape[0] == 0 or reg_it.shape[0] == 0:
        continue

       reg_it = reg_it.astype(precision)

       if it == 0:
        etype = etype_it
        nobj = nobj_it
        reg = reg_it
       else:
        etype = np.concatenate((etype, etype_it))
        nobj = np.concatenate((nobj, nobj_it))
        reg = np.concatenate((reg, reg_it))
    
       etype.astype(precision)
       reg.astype(precision)

       print ('event type', etype.dtype, etype.shape, float(etype.nbytes)/1.e6, 'MB')   
       print ('numb objects', nobj.dtype, nobj.shape, float(nobj.nbytes)/1.e6, 'MB')
       print ('regression', reg.dtype, reg.shape, float(reg.nbytes)/1.e6, 'MB')

    return etype, nobj, reg
