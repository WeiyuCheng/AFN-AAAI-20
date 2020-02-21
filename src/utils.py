import glob
import re

def get_third_nearest_checkpoint(path):
    filenames = glob.glob(path + '/model.ckpt-*.index')
    pattern = re.compile(r'model.ckpt-(.*?).index', re.S)
    versions = []
    for filename in filenames:
        versions += [int(re.findall(pattern, filename)[0])]
    versions = sorted(versions)
    return path+'/model.ckpt-'+str(versions[-3])