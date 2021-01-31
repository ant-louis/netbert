import os
import re
import string
import argparse

import pandas as pd
from tqdm import tqdm


def parse_arguments():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirpath",
                        type=str,
                        required=True,
                        help="Path to directory where RFCs are saved under '$dirpath/raw/'.",
    )
    arguments, _ = parser.parse_known_args()
    return arguments


def load_rfc_info(filepath):
    """
    """
    # Load csv file about RFCs.
    df = pd.read_csv(filepath, index_col=0)
    
    # Get info about each RFC.
    names = df['Name'].str.lower().tolist()  # Get RFC names.
    titles = df['Title'].str.lstrip().tolist()  # Get RFC titles.
    dates = df['Date'].tolist()  # Get dates of publication.
    authors = df['Authors'].tolist()  # Get authors of paper.
    
    return names, titles, dates, authors


def write_about(name, title, date, author):
    """
    """
    return "* {} - {} * The {} is about {}. It has been writtent by {}, and published in {}.".format(name, title, name, title, author, date)


def is_sentence(text):
    """
    Check if a piece of text ends by a punctuation.
    """
    return text and text[-1] in set('!.:;?')


def process_section_names(text_list):
    """
    """
    # Extract sentences from the list.
    sentences = []
    while text_list and is_sentence(text_list[-1]):
        sentences.insert(0, text_list.pop(-1))

    # Remaining text that are not sentences are sections.
    sections = text_list

    # Join sections together.
    section = '- ' + ' - '.join(sections) + ' ' if (sections and sections[0]) else ''

    # Join sentences together.
    sentence = ' '.join(sentences)
    
    # Build formatted line.
    line = section + '* ' + sentence
    return line


def process_lines(lines, name, title, date, author):
    """
    Given the lines from the raw RFC document, concat the lines 
    forming the same paragraph.
    """
    # Remove all "head" lines in above each page.
    lines = [line for line in lines if not bool(re.search("\[Page.*\]", line))]
    lines = [line for line in lines if not bool(re.search("RFC.*" + date, line))]
    
    # Create paragraphs from lines.
    new_lines = []
    section_name = []
    section_whitespaces = [0]
    chunk = ''

    while lines:

        # Pop the current line.
        line = lines.pop(0)

        # If line is empty, skip it.
        if not line: continue
        
        # Get the number of whitespaces at the beginning of the line.
        line_whitespaces = len(line) - len(line.lstrip(' '))

        # If the number of whitespaces at the beginning of the current line is bigger than the up-to-date number of whitespaces of the current (sub)section...
        if line_whitespaces > section_whitespaces[-1]:
            # It means that the previous chunk was a subsection.
            section_name.append(chunk)  # append previous chunk as a subsection to the current section names list.
            section_whitespaces.append(line_whitespaces)  # update the number of whitespaces for the current subsection.

        # If the number of whitespaces at the beginning of the current line is exactly equal to the up-to-date number of whitespaces of the current (sub)section...
        elif line_whitespaces == section_whitespaces[-1]:
            # It means that the previous chunk was not a new subsection but well a paragraph of the current section.
            sections = process_section_names(section_name.copy())
            paragraph = sections + chunk  # create new paragraph as the concatenation of the section and subsections names + the previous chunk.
            new_lines.append(paragraph)  # append it to new_lines.

        # If the number of whitespaces at the beginning of the current line is lower than the up-to-date number of whitespaces of the current (sub)section...
        else:
            # It means that we moved out of the current subsection.
            sections = process_section_names(section_name.copy())
            paragraph = sections + chunk  # create new paragraph as the concatenation of the section and subsections names + the previous chunk.
            new_lines.append(paragraph)  # append it to new_lines.

            back_steps = min(range(len(section_whitespaces)), key=lambda i: abs(section_whitespaces[::-1][i]-line_whitespaces))  # get number of steps to backward in section.
            for step in range(back_steps):
                section_name.pop(-1)  # remove the last encountered subsections from the section names list.
                section_whitespaces.pop(-1)  # update the number of whitespaces for the current subsection.

        # Get the next chunk.
        chunk = line
        while lines and lines[0]:
            chunk += lines.pop(0)  # add new line to paragraph.
            
    # Remove lines with too many spaces.
    new_lines = [line for line in new_lines if line.count(' ')/max(len(line), 1) < 0.5]
    
    # Remove all multiple spaces.
    new_lines = [re.sub('\s{2,}', ' ', line) for line in new_lines]
    
    # Remove lines with too many special characters.
    new_lines = [line for line in new_lines if len(re.sub('[A-Za-z0-9\s]+', '', line))/max(len(line) - line.count(' '), 1) < 0.35]
    
    # If line begins with a number, remove it.
    new_lines = [line.split(maxsplit=1)[1] if (len(line.split(maxsplit=1))>1 and line.split(maxsplit=1)[0][0].isdigit()) else line for line in new_lines]
    
    # Remove too long lines.
    new_lines = [line for line in new_lines if len(line) < 1500]
    
    # Add title of RFC to beginning of each chunk.
    new_lines = [ "* {} - {} {}".format(name, title, line) for line in new_lines]
    
    # Add 'about' line.
    about = write_about(name, title, date, author)
    new_lines.insert(0, about)
    
    # Concat lines from the same paragraph that were separated due to a figure or example.
    final_lines = []
    while new_lines:
        # Get current line and extract its section.
        curr_line = new_lines.pop(0)
        curr_section = curr_line.split('*', 2)[1].lstrip().rstrip()
        curr_text = curr_line.split('*', 2)[2].lstrip().rstrip()

        # Get next line (if exists) and extract its section.
        if new_lines:
            next_line = new_lines[0]
            next_section = next_line.split('*', 2)[1].lstrip().rstrip()
            next_text = next_line.split('*', 2)[2].lstrip().rstrip()

            if (
                curr_text[-1] not in set(string.punctuation) and  # if current line text doesn't end with punctuation,
                next_section == curr_section and  # and next section is the same than previous one,
                next_text[0].islower()  # and first letter of next line text is lowercase.
            ):
                new_lines.pop(0)
                curr_line += (' ' + next_text)

        # Append current line.
        final_lines.append(curr_line)
    
    return final_lines


def main(args):
    """
    """
    # Load info about all RFCs.
    names, titles, dates, authors = load_rfc_info(os.path.join(args.dirpath, 'info.csv'))

    # Process text of each RFC.
    for i, (name, title, date, author) in tqdm(enumerate(zip(names, titles, dates, authors)), total=len(names)):
        
        # Open current RFC file.
        with open(os.path.join(args.dirpath, 'raw', name + '.txt'), 'rb') as f:
            lines = [line.decode('latin1').rstrip() for line in f]
        
        # Process and clean lines.
        processed_lines = process_lines(lines, name, title, date, author)
        
        # Save processed lines.
        os.makedirs(os.path.join(args.dirpath, 'processed'), exist_ok=True)
        with open(os.path.join(args.dirpath, 'processed', name + '.txt'), 'w') as out:
            for line in processed_lines:
                out.write(str(line) + '\n')
                
                
if __name__=="__main__":
    args = parse_arguments()
    main(args)
