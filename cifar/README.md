# TTAC++ on CIFAR10/100

TTAC++ on CIFAR-10/100 under common corruptions or natural shifts. Our implementation is based on [repo](https://github.com/vita-epfl/ttt-plus-plus/tree/main/cifar) and therefore requires some similar preparation processes.


### Requirements

To install requirements:

```
pip install -r requirements.txt
```

To download datasets:

```
export DATADIR=/data/cifar
mkdir -p ${DATADIR} && cd ${DATADIR}
wget -O CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf CIFAR-10-C.tar
wget -O CIFAR-100-C.tar https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1
tar -xvf CIFAR-100-C.tar
```

### Pre-trained Models

The checkpoints of pre-train Resnet-50 can be downloaded (214MB) using the following command:

```
mkdir -p results/cifar10_joint_resnet50 && cd results/cifar10_joint_resnet50
gdown https://drive.google.com/uc?id=1TWiFJY_q5uKvNr9x3Z4CiK2w9Giqk9Dx && cd ../..
mkdir -p results/cifar100_joint_resnet50 && cd results/cifar100_joint_resnet50
gdown https://drive.google.com/uc?id=1-8KNUXXVzJIPvao-GxMp2DiArYU9NBRs && cd ../..
```

These models are obtained by training on the clean CIFAR10/100 images using semi-supervised SimCLR.


### Create Adversarial dataset for CIFAR 10.

In this repo, we provide the generating code `utils/create_cifar10_attack_data.py`, and you can run the command as follow:

```
python utils/create_cifar10_attack_data.py
```

After you generating the adversarial dataset, you can run the evaluation experiment on it by modifying the corruption type to `attack` in each script described below.

### One Pass Protocols:

- run TTAC++ on CIFAR10\100-C under the sTTT (**N-O**) protocol.

    ```
    # CIFAR10-C: 
    bash scripts/ttac2/run_ttac2_cifar10_no.sh

    # CIFAR100-C: 
    bash scripts/ttac2/run_ttac2_cifar100_no.sh
    ```

    The following results are yielded by the above scripts (classification errors) under the snow corruption:


    | Method | CIFAR10-C | CIFAR100-C |
    |:------:|:---------:|:----------:|
    |  Test  |   21.93   |    54.57   |
    |  TTAC  |   10.01   |    37.69   |
    |  TTAC++ |  8.82    |    35.59   |

- run TTAC++ on CIFAR10\100-C under the **N-O-SF** without any source information including source statistics collected from training set.
    
    **Note**: In this work, we endeavor to mitigate the dependence of previous work on source statistics from training set. We derive the approximated source domain distribution via gradient descent as implemented in `utils/find_prototypes.py`.

    ```
    # CIFAR10-C: 
    bash scripts/ttac2/run_ttac2_cifar10_no_sf.sh

    # CIFAR100-C: 
    bash scripts/ttac2/run_ttac2_cifar100_no_sf.sh
    ```

    The following results are yielded by the above scripts (classification errors) under the snow corruption:

    | Method | CIFAR10-C | CIFAR100-C |
    |:------:|:---------:|:----------:|
    |  Test  |   21.93   |    54.57   |
    |  TTAC++  |   10.95    |    39.81   |


- run TTAC++ on CIFAR10\100-C under the **Y-O** protocol.

    ```
    # CIFAR10-C: 
    bash scripts/ttac2/run_ttac2_cifar10_yo.sh

    # CIFAR100-C: 
    bash scripts/ttac2/run_ttac2_cifar100_yo.sh
    ```

    The following results are yielded by the above scripts (classification errors) under the snow corruption:

    | Method | CIFAR10-C | CIFAR100-C |
    |:------:|:---------:|:----------:|
    |  Test  |   21.93   |    54.57   |
    |  TTAC  |   9.99    |    34.97   |
    |  TTAC++  |   8.86    |    34.50   |

### Multiple Pass Protocols:

- run TTAC++ on CIFAR10\100-C under the **N-M** protocol.

    ```
    # CIFAR10-C: 
    bash scripts/ttac2/run_ttac2_cifar10_nm.sh

    # CIFAR100-C: 
    bash scripts/ttac2/run_ttac2_cifar100_nm.sh
    ```

    The following results are yielded by the above scripts (classification errors) under the snow corruption:

    | Method | CIFAR10-C | CIFAR100-C |
    |:------:|:---------:|:----------:|
    |  Test  |   21.93   |    54.57   |
    |  TTAC  |   8.80    |    34.29   |
    |  TTAC++ |  6.32    |    29.01   |

- run TTAC++ on CIFAR10\100-C under the **N-M-SF** without any source information including source statistics collected from training set.
  
    ```
    # CIFAR10-C: 
    bash scripts/ttac2/run_ttac2_cifar10_nm_sf.sh

    # CIFAR100-C: 
    bash scripts/ttac2/run_ttac2_cifar100_nm_sf.sh
    ```

    The following results are yielded by the above scripts (classification errors) under the snow corruption:

    | Method | CIFAR10-C | CIFAR100-C |
    |:------:|:---------:|:----------:|
    |  Test  |   21.93   |    54.57   |
    |  TTAC++ |  6.32    |    34.85   |



- run TTAC++ on CIFAR10\100-C under the **Y-M** protocol.

    ```
    # CIFAR10-C: 
    bash scripts/ttac2/run_ttac2_cifar10_ym.sh

    # CIFAR100-C: 
    bash scripts/ttac2/run_ttac2_cifar100_ym.sh
    ```

    The following results are yielded by the above scripts (classification errors) under the snow corruption:

    | Method | CIFAR10-C | CIFAR100-C |
    |:------:|:---------:|:----------:|
    |  Test  |   21.93   |    54.57   |
    |  TTAC  |   8.00    |    30.48   |
    |  TTAC++ |  6.94    |    27.93   |


### Descriptions

- Both `TTAC2_multipass.py` and `TTAC2_multipass2.py` are the implementations of TTAC++ under multiple pass protocol, except that `TTAC2_multipass.py` will be more memory efficient but slower, while `TTAC2_multipass2.py` will be faster. 
- `TTAC2_onepass.py` and `TTAC2_onepass2.py` are similar situation. 


### Acknowledgements

Our code is built upon the public code of the [TTT++](https://github.com/vita-epfl/ttt-plus-plus/tree/main/cifar) and [TTAC](https://github.com/Gorilla-Lab-SCUT/TTAC).