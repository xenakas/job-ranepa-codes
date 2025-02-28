from utils import get_table_ids
from parse_goszakupki_auto_ru_2019 import *


ids = get_table_ids(sql_write)
files = query_all_files(sql)
files = sorted([f for f in files if f[0] in ids])
