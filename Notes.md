# Notes

## Set up servers

- @zapp (144.254.7.96)
    - ssh antoloui@144.254.7.96
    - cd /raid/antoloui
    - Data: cd /raid/dgoloube
        - Cisco : /cdc
        - Wikipedia : /wiki/extractor/tekst/AA/combined

- @fullfat (10.48.56.67) : 3.9 TeraBytes
    - Jupyter Lab
        - ssh -L 8888:localhost:9000 antoloui@10.48.56.67
        - conda activate bert
        - cd /raid/antoloui/Master-thesis
        - jupyter lab --no-browser --port=9000 --ip=0.0.0.0

    - Tensorboard
        - ssh -L 16006:localhost:6006 antoloui@10.48.56.67
        - conda activate bert
        - cd /raid/antoloui/Master-thesis/Code/bert_singleGPU
        - tensorboard --logdir=./pretraining_output_fullfat

- @megafat (10.48.56.75) : 3.3 TeraBytes
    - Jupyter Lab
        - ssh -L 8080:localhost:9090 antoloui@10.48.56.75
        - conda activate bert
        - cd /raid/antoloui/Master-thesis
        - jupyter lab --no-browser --port=9090 --ip=0.0.0.0

## Detach processes

- Create new session: screen -S name
- Detach from this session: ctrl+a+d
- Re-attach to this session: screen -r name
- List the sessions: screen -ls


## Size of data

* Number of words: 2 636 669 160 (2.6B)
* Vocabulary size (unique words): 3 810 094 (3.8M)

## Create pretraining data

- 7s for 102 836 words
- 53.4 hours for 2.6B words -> 2,2 jours

## Training

### Pretraining data
- Instances:
    + 1: 22 323 943 total instances (17.1765 GB)
    + 2: 1 688 007 total instances (1.2934 GB)
    + 3: 4 536 042 total instances (3.4834 GB)
    + 4: 38 465 591 total instances (29.4735 GB)
    + 5: 14 009 338 total instances (10.7820 GB)
    + 6: 12 672 207 total instances (9.7201 GB)
    + 7: 11 127 002 total instances (8.5420 GB)
    + 8: 13 775 060 total instances (10.5401 GB)
    + 9: 9 287 038 total instances (7.1263 GB)
    + 10: 11 571 008 total instances (8.8751 GB)
    + 11: 28 050 132 total instances (21.5082 GB)
    + 12: 39 272 210 total instances (30.1326 GB)
    + 13: 10 687 total instances (0.0083 GB)
    + TOTAL: 206 788 265 (158.6615 GB) 
    
## Iterations
    
- 1 epoch: 
    * batch=32: 6 462 133 iterations
    * batch=64: 3 231 066 iterations
