import os
import re
import argparse

import wget
import requests
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_arguments():
    """
    url, errors_filepath, outdir
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_url",
                        type=str,
                        default='https://tools.ietf.org/rfc/index',
                        help="URL of the online RFC index.",
    )
    parser.add_argument("--rfc_base_url",
                        type=str,
                        default='https://tools.ietf.org/rfc/', 
                        help="Base URL for downloading a RFC in text form.",
    )
    parser.add_argument("--outdir",
                        type=str,
                        required=True,
                        help="Output directory.",
    )
    arguments, _ = parser.parse_known_args()
    return arguments



def concat_rfc_lines(lines):
    """
    Given a list of lines where a same RFC is described on multiple lines, concat
    the lines describing the same RFC.
    """
    rfc_lines = []
    current_rfc = ''
    for line in lines:
        if line.startswith('RFC'):
            rfc_lines.append(current_rfc)  # End of previous RFC, append it to list.
            current_rfc = line  # Get beginning of new rfc.
        else:
            current_rfc += line
    return rfc_lines


def remove_multiple_spaces(text):
    """
    Given a string, replace all multiple spaces in it by a single space.
    """
    text = re.sub('\s{2,}', ' ', text)
    text = text.lstrip().rstrip()  # Remove whitespaces in beginning or end of string.
    return text
    

def get_rfc_lines(page):
    """
    Given the result of an url request, get the text of interest.
    """
    # Load the page with BeautifulSoup.
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # Get the text of interest (the index is in <pre>...</pre>).
    body = soup('pre')[0]
    
    # Get plain text.
    content = body.get_text() 
    
    # Remove all text before the line beginning by 'RFC1' (beginning of the index).
    content = content.split('RFC1 ')[1]
    content = 'RFC1 ' + content
    
    # Split raw text to lines.
    lines = content.splitlines()
    lines = [line for line in lines if line != ''] # remove empty lines.
    
    # Concat lines describing the same RFC.
    rfc_lines = concat_rfc_lines(lines)
    rfc_lines = rfc_lines[1:]
    
    # Remove multiple spaces in lines.
    rfc_lines = [remove_multiple_spaces(line) for line in rfc_lines]
    
    # Remove all 'Non Issued' RFC lines.
    rfc_lines = [line for line in rfc_lines if 'Not Issued' not in line]
    
    return rfc_lines


def create_dataframe(rfc_lines):
    """
    Given the lines describing each RFC, create a dataframe.
    """
    # Init lists.
    names = []
    titles = []
    authors = []
    dates = []
    formats = []
    obsolotes = []
    obsoloted = []
    updates = []
    updated = []
    also = []
    status = []
    dois = []
    
    # Process each line.
    for i, line in enumerate(tqdm(rfc_lines)):
        
        # Get all attributes within brackets.
        brackets = re.findall(r"\((.*?)\)", line)

        # Get individual attributes.
        form = None
        obs = None
        obs_by = None
        up = None
        up_by = None
        al = None
        stat = None
        doi = None
        for att in brackets:
            if att.startswith('Format: '):
                form = att.split('Format: ')[1]
            elif att.startswith('Obsolotes '):
                obs = att.split('Obsolotes ')[1]
            elif att.startswith('Obsoleted by '):
                obs_by = att.split('Obsoleted by ')[1]
            elif att.startswith('Updates '):
                up = att.split('Updates ')[1]
            elif att.startswith('Updated by '):
                up_by = att.split('Updated by ')[1]
            elif att.startswith('Also '):
                al = att.split('Also ')[1]
            elif att.startswith('Status: '):
                stat = att.split('Status: ')[1]
            elif att.startswith('DOI: '):
                doi = att.split('DOI: ')[1]
        line = line.split('(Format')[0].rstrip()  # Remove bracket attributes from the line.

        # Get the date of publication.
        split_line = line.split(".")
        split_line = [l for l in split_line if l != '']
        date = split_line[-1].lstrip()
        line = line.replace(date + '.', '')  # Remove date from line.

        # Get name of RFC.
        name = line.split()[0]
        line = line.replace(name, '')  # Remove name from line.

        # Get title of RFC.
        split_line = line.split('.')
        title = split_line.pop(0)
        while not split_line[0].isspace() and not (len(split_line[0]) == 2 and split_line[0][0].isspace() and split_line[0][1].isupper()):
            title += ('.' + split_line.pop(0))  # This line deals with a title that contains dots.
        line = line.replace(title + '.', '')  # Remove title from line.

        # Get authors.
        aut = line.lstrip().rstrip()[:-1]

        # Append all info to corresponding list.
        names.append(name)
        titles.append(title)
        authors.append(aut)
        dates.append(date)
        formats.append(form)
        obsolotes.append(obs)
        obsoloted.append(obs_by)
        updates.append(up)
        updated.append(up_by)
        also.append(al)
        status.append(stat)
        dois.append(doi)

    # Create dataframe.
    d = {'Name':names,
         'Title':titles,
         'Authors':authors,
         'Date':dates,
         'Formats':formats,
         'Obsolotes':obsolotes,
         'Obsoloted_by':obsoloted,
         'Updates':updates,
         'Updated_by':updated,
         'Also_FYI':also,
         'Status':status,
         'DOI':dois}
    df = pd.DataFrame(d)
    return df


def download_all(rfc_base_url, rfc_ids, outdir):
    """
    """
    os.makedirs(outdir, exist_ok=True)  # Create output directory if not exists.

    errors = []  # Keep track of rfc badly downloaded.
    for rfc in tqdm(rfc_ids):
        url = rfc_base_url + rfc + '.txt'  # Create the download url.
        try:
            wget.download(url, outdir)
        except Exception as e:
            errors.append(rfc)
            print("{}: HTTP Error 404 - Not Found.".format(rfc))
            
    return errors
        

def main(args):
    """
    """
    print("\nDownload the index page at {}...".format(args.index_url))
    page = requests.get(args.index_url)
    
    print("\nExtract all RFC lines...")
    rfc_lines = get_rfc_lines(page)
    
    print("\nProcess index lines...")
    df = create_dataframe(rfc_lines)
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, 'info.csv'), sep=',', encoding='utf-8', float_format='%.10f', decimal='.')
    
    print("\nDownload all RFC files to {}...".format(os.path.join(args.outdir, 'raw')))
    rfc_ids = df['Name'].str.lower().tolist()
    errors = download_all(args.rfc_base_url, rfc_ids, os.path.join(args.outdir, 'raw'))
    print("Download errors: {}".format(str(errors)))
    print("\nDONE.")


if __name__=="__main__":
    args = parse_arguments()
    main(args)
