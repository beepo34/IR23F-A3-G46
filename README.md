# IR23F-A3-G46
CS 121 - Assignment 3: Search Engine

## Team Members
Teresa Liang @teresa-liang
- NetID: liangth1
- Student ID: 94351677

Timothy Wada @wadatimothy
- NetID: tjwada
- Student ID: 89801556

Aileen Mi @beepo34
- NetID: mia1
- Student ID: 15166075 

## Configuration

To install the dependencies for this project run the following command after ensuring pip is installed for the version of python you are using. Admin privileges might be required to execute the commands. Also make sure that the terminal is at the root folder of this project.

```
python -m pip install -r packages/requirements.txt
```

To set up the data files for this project unzip the contents of the `developer.zip` file from the Assignment 3: M1 Canvas assignment into the root folder of this project. 

## Building the Index

To start the code to build the inverted index, run the following command.

```
python3 index.py
```

## Starting the Search Engine Using the Console

To start the search engine on a console interface, run the following command.

```
python3 query.py
```

The console will repeatedly prompt the user to enter a query. Enter `exit` to quit the engine.

## Starting the Search Engine Using the GUI Interface
To launch the the search engine application with GUI interface, run the following command.

```
python main.py
```
