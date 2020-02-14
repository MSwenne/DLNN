# DLNN
> This is the first Deep Learning and Neural Network assignment. 
>This repository is maintained by:
>* 	Martijn Swenne
>* 	Vince Hasse
>* 	Orson Peters
>* 	Amin Moradi

@All rights  reserved.

## Python virtual env

You can create a virtual env in the source of the project to prevent any version conflicts between other
environments and packages. 

Make sure you have `virtualenv` install on your machine. 
You can install it via:

```
# For mac/linux
python3 -m pip install --user virtualenv

# Windows 
py -m pip install --user virtualenv

```

## Create Assignment 0 Virtual Env
You can checkout all the required packages for this assignment in `requirements.txt`.
To create a new virtual env at the `ROOT` of the project run:
```
python3 -m venv env_dl_0
```
This will create a `env_dl_0` folder at the root of the project for you which as well must be in .gitignore file.

### Activate virtual env
To activate the install virtual env you can simply run: 
```
# For mac
source env_dl_0/bin/activate

# For windows
.\env_dl_0\Scripts\activate

```

### Install packages
To install all the packages in `requirements.txt` you can run: 

```
# For mac/linux
pip install -r requirements.txt

# For windows
python -m pip install -r requirements.txt

```


### Updating requirements.txt
After you add or remove a package to your virtual env, please make sure you update the `requirements.txt`.
You can simply run:

```
pip freeze > requirements.txt
```
