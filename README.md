# Pynama
Fem problem solver with PETSc tools

### Necessary Installation: 

- python3
- mpi4py
    - `pip install mpi4py`
- numpy
    - `pip install numpy`
- matplotlib
    - `pip install matplotlib`
- petsc 
    - clone petsc   `git clone --branch maint-3.12 [https://gitlab.com/petsc/petsc.git](https://gitlab.com/petsc/petsc.git)`
    - cd pestc
    - configuration `./configure --download-hdf5 --download-chaco --download-fblaslapack`
    - make 
- petsc4py 
    - clone petsc4py `git clone --branch maint-3.12 [https://gitlab.com/petsc/petsc4py.git](https://gitlab.com/petsc/petsc4py.git)`
    - change in setup.cfg  dir and arch from pestc
        - like this config
            ``` 
            petsc_dir = /home/name/petsc
            petsc_arch = /arch-python-linux-x86_64
                ``` 

    - `python3 setup.py build -v`
    - `python3 setup.py install --user`

