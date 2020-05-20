import os
import argparse

import re
import ftfy
from tika import parser

import nltk
from nltk.tokenize import sent_tokenize


def parse_arguments():
    """
    Parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile",
                        "-i",
                        default='/raid/antoloui/Master-thesis/Data/QA/Cisco_CCNA.pdf',
                        type=str,
                        help="The input file from which to extract text.",
    )
    parser.add_argument("--outdir",
                        "-o",
                        default=None,
                        type=str,
                        help="The output directory.",
    )
    arguments, _ = parser.parse_known_args()
    return arguments


def load_pdf(filepath):
    """
    Given path of pdf file, extract content.
    """
    raw = parser.from_file(filepath)
    text = raw['content']
    return text


def general_cleaning(text):
    """
    General cleaning for pdf.
    """
    text = ftfy.fix_text(text)  #Fix text with ftfy
    text = text.replace('\n', ' ')  #Replace all '\n' by a space.
    text = text.replace('- ', '')  #Remove all liaisons between split words.
    text = re.sub('\s{2,}', ' ', text)  #Replace two or more spaces with one
    return text


def custom_cleaning(text):
    """
    Cleaning specific to this particular pdf.
    """
    text = re.sub(r'\bptg\d*\b', '', text)  #Remove all weird '\ptgxxxx' tokens.
    
    # Remove name of the book and all chapter names above each page of the pdf.
    to_remove = ['CCENT/CCNA ICND1 100-105 Official Cert Guide',
                 'Introduction', 
                 'Your Study Plan', 
                 'Chapter 1: Introduction to TCP/IP Networking', 
                 'Chapter 2: Fundamentals of Ethernet LANs',
                 'Chapter 3: Fundamentals of WANs',
                 'Chapter 4: Fundamentals of IPv4 Addressing and Routing',
                 'Chapter 5: Fundamentals of TCP/IP Transport and Applications',
                 'Chapter 6: Using the Command-Line Interface',
                 'Chapter 7: Analyzing Ethernet LAN Switching',
                 'Chapter 8: Configuring Basic Switch Management',
                 'Chapter 9: Configuring Switch Interfaces',
                 'Chapter 10: Analyzing Ethernet LAN Designs',
                 'Chapter 11: Implementing Ethernet Virtual LANs',
                 'Chapter 12: Troubleshooting Ethernet LANs',
                 'Chapter 13: Perspectives on IPv4 Subnetting',
                 'Chapter 14: Analyzing Classful IPv4 Networks',
                 'Chapter 15: Analyzing Subnet Masks',
                 'Chapter 16: Analyzing Existing Subnets',
                 'Chapter 17: Operating Cisco Routers',
                 'Chapter 18: Configuring IPv4 Addresses and Static Routes',
                 'Chapter 19: Learning IPv4 Routes with RIPv2',
                 'Chapter 20: DHCP and IP Networking on Hosts 470',
                 'Chapter 21: Subnet Design',
                 'Chapter 22: Variable-Length Subnet Masks',
                 'Chapter 23: IPv4 Troubleshooting Tools',
                 'Chapter 24: Troubleshooting IPv4 Routing',
                 'Chapter 25: Basic IPv4 Access Control Lists',
                 'Chapter 26: Advanced IPv4 Access Control Lists',
                 'Chapter 27: Network Address Translation',
                 'Chapter 28: Fundamentals of IP Version 6',
                 'Chapter 29: IPv6 Addressing and Subnetting',
                 'Chapter 30: Implementing IPv6 Addressing on Routers',
                 'Chapter 31: Implementing IPv6 Addressing on Hosts',
                 'Chapter 32: Implementing IPv6 Routing',
                 'Chapter 33: Device Management Protocols',
                 'Chapter 34: Device Security Features',
                 'Chapter 35: Managing IOS Files',
                 'Chapter 36: IOS License Management',
                 'Chapter 37: Final Review']
    for s in to_remove:
        text = text.replace(s, '')
        
    return text


def split_sentences(text):
    """
    Split text into 1 sentence per line.
    """
    sent_list = sent_tokenize(text)
    return '\n'.join(sent_list)


def main(args):
    print("Extract content from {}...".format(args.infile))
    text = load_pdf(args.infile)
    print("  Done.")
    
    print("Perform a first general cleaning...")
    text = general_cleaning(text)
    print("  Done.")
    
    print("Perform a second custom cleaning...")
    text = custom_cleaning(text)
    print("  Done.")
    
    print("Split text into sentences...")
    text = split_sentences(text)
    print("  Done.")
    
    if args.outdir is None:
        args.outdir = os.path.dirname(args.infile)
    outfile = os.path.splitext(os.path.basename(args.infile))[0] + '.txt'
    outpath = os.path.join(args.outdir, outfile)
    
    print("Save extracted text to {}...".format(outpath))
    with open(outpath, "w") as out:
        out.write(text)
    print("  Done.")
    

if __name__=="__main__":
    args = parse_arguments()
    main(args)
