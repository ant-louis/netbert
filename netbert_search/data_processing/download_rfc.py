import os
import wget
import pickle
from tqdm import tqdm


def main(base_url, outdir, total_rfc):
    """
    """
    # Create output directory if not exists.
    os.makedirs(outdir, exist_ok=True)

    # Download all RFCs.
    errors = []
    for i in tqdm(range(total_rfc)):
        # Create the url.
        url = base_url + str(i+1) + '.txt'

        # Download page.
        try:
            wget.download(url, outdir)
        except Exception as e:
            errors.append(i+1)
            print("RFC {}: HTTP Error 404 - Not Found.".format(i+1))
            
    # Save 404 errors.
    with open(os.path.join(outdir, 'errors404'), 'wb') as out:
        pickle.dump(errors, out)
    

if __name__=="__main__":
    main(base_url='https://tools.ietf.org/rfc/rfc', 
         outdir='/raid/antoloui/Master-thesis/_data/search/rfc', 
         total_rfc=8774)
