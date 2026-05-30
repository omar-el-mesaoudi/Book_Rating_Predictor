# Book_Rating_Predictor
Work in progress.

omar/
├── Dataset/
│   └── books.csv (The data is now cleaned from the original project's file)
│
├── Notebook/
│   └── clean_csv.ipynb (This notebook is to run only once to clean part of the CSV to make the file readable)
│   └── main.ipynb (This is the main notebook, including the data exploration, analysis, machine learning, etc. parts)
│   └── Dockerfile
│   └── inference.py (It was copied into the image when it wass builded)
│   └── test_df.csv (It is necessary in order to run the container from the DockerHub)
|
└── README.md
└── docker_steps.md
└── requirements.txt
└── copy_csv.py (to copy the raw data set CSV to the CSV that we will work on, so that we will always have the original csv file)
