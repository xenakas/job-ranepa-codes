import io
import os
import re
import shutil
import logging
import traceback
from typing import List, Optional, Union

from glob import glob
from itertools import chain
import subprocess

import pandas as pd

import camelot
import textract
from pyunpack import Archive
from unidecode import unidecode
from google_images_download import google_images_download
from docx.api import Document
from docx.opc.exceptions import PackageNotFoundError
from .pdf import PDFProcessing
# from .mwd import DocXProcessing
from .proxy_ferret import ferret
from .utils import check_false_ord

# from utils_d.pdf import PDFProcessing
# from utils_d.mwd import DocXProcessing
# from utils_d.proxy_ferret import ferret
# from utils_d.utils import check_false_ord

logger = logging.getLogger(__name__)
pdf = None
# docx = DocXProcessing(logger)

proxer = None

logging.basicConfig(level=logging.DEBUG, filename='message.log')

google_img_response = None

valid_extensions = ['csv', 'doc', 'docx', 'eml', 'epub', 'gif', 'htm', 'html',
                    'jpeg', 'jpg', 'json', 'log', 'mp3', 'msg', 'odt', 'ogg',
                    'pdf', 'png', 'pptx', 'ps', 'psv', 'rtf', 'tff', 'tif',
                    'tiff', 'tsv', 'txt', 'wav', 'xls', 'xlsx']
process_file_extensions = ["pdf", "odt", "doc"]
text_formats = {'csv', 'doc', 'docx', 'eml', 'epub', 'htm', 'html',
                'log', 'msg', 'odt',
                'pdf', 'pptx', 'psv', 'rtf', 'tff',
                'tsv', 'txt', 'xls', 'xlsx'}
process_file_extensions = ["pdf", "odt", "doc"]


def link2text(link, attempts=0, folder_name="", max_attempts=100,
              to_delete=True):
    global proxer
    if not proxer:
        proxer = ferret.Ferret()
    texts = []
    if link.startswith("www"):
        link = "http://" + link
    link = link.replace("##slash##", '/')
    print("link", link, attempts)
    response = proxer.get_url_with_proxy(link, attempt=attempts)
    filename_extension = ""
    try:
        response.ok
        r = response.content
        if 'content-disposition' in response.headers:
            filename = response.headers['content-disposition']
            filename = re.findall("filename=\"(.+)\"", filename)
            if filename:
                filename = filename[0]
                filename_extension = filename.split(".")[-1]
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        r = None
    if response and r:
        if not filename_extension or\
                filename_extension not in process_file_extensions:
            if str(r[:10]).startswith("b'%PDF"):
                filename_extension = "pdf"
            elif str(r[:10]).startswith("b'\\xd0\\xcf"):
                filename_extension = "doc"
            elif str(r[:10]).startswith("b'Rar"):
                filename_extension = "rar"
            elif str(r[:10]).startswith("b'PK"):
                filename_extension = "docx"
            elif str(r[:10]).startswith(r"b'\xff\xd8"):
                filename_extension = "jpeg"
            elif str(r[:10]).startswith("b'{\\rtf1"):
                filename_extension = "rtf"
        filename = "file.{}".format(filename_extension)
        if folder_name:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
        # if folder name is empty, bool(folder_name) * "/" is also empty
        filename = folder_name + bool(folder_name) * "/" + filename
        if filename_extension in valid_extensions:
            texts = [process_file(filename, r,
                                  to_delete=to_delete,
                                  folder_name=folder_name)]
        elif filename_extension == "docx":
            try:
                r_bytes = io.BytesIO(r)
                text = list(docx.page(r_bytes))
                # text = [' '.join(t.strip().split()) for t in text]
                # text = ' '.join(text)
                # texts = [text]
            except Exception as ex:
                filename_extension = "odt"
                filename = "file.{}".format(filename_extension)
                filename = folder_name + bool(folder_name) * "/" + filename
            texts = [process_file(filename, r, to_delete=to_delete,
                                  folder_name=folder_name)]
        elif filename_extension in ["rar", "zip"]:
            archive_file = filename
            with open(archive_file, "wb") as f:
                f.write(r)
            if '__file__' in globals():
                base_path = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__)))
            else:
                base_path = ""
            unpack_dir = os.path.join(base_path, folder_name)
            unpack_dir = os.path.join(unpack_dir, "unpack_tmp")
            if not os.path.exists(unpack_dir):
                os.makedirs(unpack_dir)
            try:
                Archive(archive_file).extractall(unpack_dir)
            except Exception as ex:
                print(ex)
                traceback.print_exc()
                return texts
            all_files = glob(unpack_dir + "**", recursive=True)
            all_files = sorted(all_files, key=len, reverse=True)
            for f_ind, fi in enumerate(all_files):
                # Rename file to include only latin and numbers and .
                if not fi.strip():
                    continue
                head, tail = os.path.split(fi)
                tail = tail.replace(" ", "_")
                try:
                    # zip формат понимает только cp437 и utf-8 кодировки
                    # На практике, были распространены архивы, которые
                    # используют OEM code page для установленного варианта
                    # Windows. Для русской Винды это cp866.
                    tail = tail.encode('cp437').decode("cp866")
                except Exception as ex:
                    print(ex, tail)
                    traceback.print_exc()
                    tail = unidecode(tail)
                tail = re.sub("[^a-zA-Zа-яА-Я/_\.0-9]", "", tail)
                if os.path.isdir(fi):
                    tail = tail.replace(".", "")
                elif "." in tail:
                    tail = tail.split(".")
                    tail = "{}.{}".format("_".join(tail[:-1]), tail[-1])
                if not tail:
                    tail = "somefile_{}".format(f_ind)
                new_name = os.path.join(base_path, head, tail)
                fi = os.path.join(base_path, fi)
                try:
                    shutil.move(fi, new_name)
                    all_files[f_ind] = new_name
                except Exception as ex:
                    print(ex)
                    traceback.print_exc()
                    print(fi, base_path, new_name)
            all_files = glob(unpack_dir + "**", recursive=True)
            folders = [fi for fi in all_files if os.path.isdir(fi)]
            folders = sorted(folders, key=len, reverse=True)
            files = [fi for fi in all_files if fi not in folders]
            for f in files:
                # File path is absolute?
                text = process_file(f, None, to_delete=to_delete,
                                    folder_name=folder_name)
                texts.append(text)
            folders = sorted(folders, key=len, reverse=True)
            if to_delete:
                [shutil.rmtree(f) for f in folders]
        else:
            print(str(r[:10]), "UNKNOWN FILE")
    # elif attempts < max_attempts:
    #     attempts += 1
    #     texts = link2text(link, attempts=attempts, folder_name=folder_name,
    #                       max_attempts=max_attempts, to_delete=to_delete)
    # return text
    return texts


def clean_docx_table(table):
        # table_name = ""
        data = list()
        columns = []
        for i, row in enumerate(table.rows):
            text = [cell.text.lower() for cell in row.cells]
            if not text:
                continue
            if i == 0:
                if "таблица" in text[0].lower():
                    # table_name = text[0]
                    continue
                elif len(text) > 1:
                    columns = text
                    continue
            text = [t.split("\n") for t in text]
            text = [[l for l in t if l] for t in text]
            text_keys_intersect = set()
            for t_i, t in enumerate(text):
                t = [re.split(" |:|-", l) for l in t]
                text[t_i] = t
                # sets union
                if t_i == 0:
                    text_keys_intersect = set([l[0] for l in t])
                else:
                    text_keys_intersect = set([l[0] for l in t]) &\
                        text_keys_intersect
            for t_i, t in enumerate(text):
                index = []
                i_t = 0
                for l_i, l in enumerate(t):
                    if l[0] in text_keys_intersect:
                        text[t_i][l_i] = " ".join(l[1:]).strip()
                        index.append(l[0])
                    else:
                        text[t_i][l_i] = " ".join(l).strip()
                        index.append(i_t)
                        i_t += 1
                text[t_i] = dict(zip(index, text[t_i]))
            all_keys = set([w for t in text for w in t.keys()])
            for t_i, t in enumerate(text):
                for k in all_keys:
                    if k not in t.keys():
                        text[t_i][k] = None
            data.append(text)
        df = pd.DataFrame(list(chain.from_iterable(data)))
        df = df.T
        if columns and len(df.columns) == len(columns):
            df.columns = columns
        return df


def get_tables_docx(filename, folder_name="./", convert=False):
    r"""
    extracting tables from docx files
    """
    if convert and not filename.endswith("docx"):
        filename = libreoffice_convert(
            filename, folder_name, out_format="docx")
    if not filename:
        return []
    try:
        document = Document(filename)
    except PackageNotFoundError:
        return []
    tables = document.tables
    processed_tables = []
    for table in tables:
        t = [[c.text for c in row.cells] for row in table.rows]
        processed_tables.append(pd.DataFrame(t))
    return processed_tables


def libreoffice_convert(filename: str, folder_name: str, out_format="pdf"):
    try:
        subprocess.call(['libreoffice', '--headless', '--convert-to',
                         out_format,
                         '--outdir', folder_name, filename])
        out_file = filename.split("/")[-1]
        out_file = re.sub("\..+$", f".{out_format}", out_file)
        out_file = os.path.join(folder_name, out_file)
    except Exception as ex:
        print("libreoffice conversion failed", filename)
        print(ex)
        traceback.print_exc()
        out_file = None
    return out_file


def get_tables_camelot(filename: str, folder_name="", to_delete=True):
    """
    extract tables using camelot; first it converts the file to .pdf
    """
    if not filename.endswith("pdf"):
        pdf_file = libreoffice_convert(filename, folder_name, out_format="pdf")
    else:
        pdf_file = filename
    print("\n".join([filename, folder_name, pdf_file]))
    tables = []
    if pdf_file and os.path.exists(pdf_file):
        try:
            tables = camelot.read_pdf(pdf_file, pages='1-end')
            tables = [t.df for t in tables]
        except Exception as ex:
            print(filename, pdf_file)
            print(ex)
            traceback.print_exc()
            return []
        if to_delete:
            os.remove(pdf_file)
    return tables


def table_scrapper(text: str) -> List[pd.DataFrame]:
    """
    simplistic table scraper
    """
    lines = text.split("\n")
    lines = [l for l in lines if "|" in l]  # "\t" in l or
    lines = [l for l in lines if len(re.findall("\|", l)) > 2]
    lines = [[w.strip() for w in l.split("|") if w.strip()] for l in lines]
    if not lines:
        return []
    max_len = max((len(l) for l in lines))
    lines = [l for l in lines if len(l) == max_len]
    if len(lines) > 1:
        return [pd.DataFrame(lines)]


def getting_tables(text, filename, filename_extension, to_delete, folder_name):
    tables = []
    if filename_extension in text_formats:
        try:
            tables = get_tables_docx(
                filename, folder_name=folder_name, convert=True)
        except Exception as ex:
            print("docx error", ex)
            tables = get_tables_camelot(filename, to_delete=to_delete,
                                        folder_name=folder_name)
    if not tables and "|" in text:
        filename = filename.split(".")
        if len(filename) > 1:
            filename = "_".join(filename[:-1]) + ".txt"
        else:
            filename = filename[0] + ".txt"
        with open(filename, "w") as f:
            f.write(text)
        tables = get_tables_camelot(filename, to_delete=to_delete,
                                    folder_name=folder_name)
        if not tables:
            tables = table_scrapper(text)
    return tables


def process_file(filename, r, get_tables=True, to_delete=True,
                 folder_name="./"):
    global pdf
    if not pdf:
        pdf = PDFProcessing(logger)
    filename_extension = filename.split(".")
    if len(filename_extension) > 1 and\
            filename_extension[-1] not in valid_extensions:
        return ""
    else:
        filename_extension = filename_extension[-1]
    if not r:
        with open(filename, "rb") as f:
            r = f.read()
    else:
        with open(filename, "wb") as f:
            f.write(r)
    text = ""
    try:
        text = textract.process(filename, language="rus").decode("utf-8")
    except Exception as ex:
        print(ex)
        print(text)
        traceback.print_exc()
        return ""

    tables = []
    if get_tables:
        tables = getting_tables(
            text, filename, filename_extension, to_delete, folder_name)
    # text = ' '.join(text.strip().split())
    try:
        if to_delete:
            os.remove(filename)
    except FileNotFoundError:
        pass
    # ?? check if there is text in pdf
    if all(check_false_ord(l, latin=True) for l in text[:10]) or \
            len(text.split()) < 10 and filename.endswith(".pdf"):
        # pdf.page is a generator, that's why we turn it to a list
        try:
            text = list(pdf.page(r))
        except RuntimeError:
            text = ""
        if text:
            text = text[0]
        if not text:
            text = ""
    if get_tables:
        text = [text, tables]
    return text


def download_image(place:str, delete=True, logo=True, site=None,
                   read_image=False) -> Union[List, None]:
    global google_img_response
    if not google_img_response:
        google_img_response = google_images_download.googleimagesdownload()
    if logo:
        logo_str = " лого"
        image_type = "clipart"
    else:
        logo_str = ""
        image_type = "photo"
    image_arguments = {
        "keywords": place + logo_str,
        "limit": 1,
        "print_urls": True,
        "type": image_type,
        "language": "Russian",
        "prefix": "company_logo_",
        "extract_metadata": True,
        "specific_site": site,
        "format": "png"}
    image_read = None
    try:
        google_img_response.download(image_arguments)
        images = glob("downloads/*/company_logo_*")
    except AttributeError:
        images = None
    if images:
        if read_image:
            image_file = open(images[0], 'rb')
            image_read = image_file.read()
            image_file.close()
            if delete:
                files = glob("downloads/**", recursive=True)[1:][::-1]
                for f in files:
                    try:
                        os.remove(f)
                    except Exception as ex:
                        print(ex)
                        os.rmdir(f)
                        traceback.print_exc()
        else:
            return images
    return image_read
