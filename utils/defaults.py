import os
import socket

from munch import munchify

hostname = socket.gethostname()

DEFAULTS = dict()

DEFAULTS['project_name'] = 'ccraft'


DEFAULTS['data_root'] = os.environ.get('CCRAFT_DATA_ROOT', '/path/to/ccraft_data')
DEFAULTS['aux_data'] = os.path.join(DEFAULTS['data_root'], 'aux_data')
DEFAULTS['project_dir'] = os.environ.get('CCRAFT_PROJECT_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULTS['experiment_root'] = os.path.join(DEFAULTS['data_root'], 'experiments')

DEFAULTS['CMU_root'] = os.environ.get('CCRAFT_CMU_ROOT', '/path/to/AMASS/smpl/CMU')

DEFAULTS = munchify(DEFAULTS)
