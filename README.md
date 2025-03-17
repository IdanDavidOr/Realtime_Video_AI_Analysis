# Real-Time Video Analysis 'ןאי YOLO
## Submission by: Idan David Or Lavi

---

## Installation - anaconda is needed for this step

Open anaconda terminal and navigate to the directory this file is in

(Optional) New anaconda environment:
   1. Create a new env using `conda create --name myenv`

   2. activate the new env using `conda activate myenv`

You can change `myenv` to any name you'd like 

### Install modules
To install the required modules run `conda install -c conda-forge --file requirements.txt`


## Using for inference

In any command line terminal, go to the folder this file is in.

Run this command: `python main.py rtsp://your_rtsp_url_here` with your rtsp url


---
## Assumptions

1. The stream is live
2. There is internet connection

---
## Pipeline design decisions

I chose YOLOv5 for ease of implementation. 

`ultralytics` module didn't easily install with conda, so I chose to use YOLOv5 for the POC.

The pipeline was incrementally designed.
Each function takes care of one aspect of the task. 

The json saving scheme uses global variables for inter-function communication

The frame is resized so that the input size is constant.


---
## Q&A

1.Q. How would you improve your code to handle multiple stream sources?

1.A. I'd create a tread for each stream, 
change the data saving scheme to take several streams in account, 
add stream managing functionality
change i/o to fit the new scheme


2.Q. What is the advantage of using a GPU for inference? Can the GPU use for decoding as well?

2.A. Speed, Parallel Processing and Scalability. a GPU can sometimes be used for decoding (), but it doesn't use its CUDA cores for it.



3.Q. For each following SDK/Python Package please write a short sentence describing it’s purpose and use cases, only if you are familiar with it. If you are familiar with it, describe how would it be relevant to your implementation

3.A.

a. asyncio:
asyncio is used for writing single-threaded concurrent code using coroutines, multiplexing I/O access over sockets and other resources, running network clients and servers, and other related primitives. 
It is used for asynchronous programming, allowing efficient handling of I/O-bound operations and enabling concurrent execution. 
This can be a different approach to threads.

b. GStreamer:
GStreamer allows the construction of graphs for processing multimedia data. 
It supports various audio and video formats, and it's used for tasks like media playback, recording, editing, streaming, and more. 
In the context of the implementation, GStreamer is used for efficient video decoding and streaming as an opencv backend.

c. Nvidia DeepStream SDK:
NVIDIA DeepStream SDK is a platform for building AI-powered video analytics applications. 
It provides a framework for processing and analyzing video streams in real-time, leveraging NVIDIA GPUs for accelerated inferencing using deep learning models. 
In the given implementation, leveraging DeepStream could significantly enhance video processing and analysis capabilities, especially if NVIDIA GPUs are available.

d. Docker:
Docker is a platform that allows you to package, distribute, and run applications in containers. 
Containers provide a lightweight, portable, and consistent environment for running applications and their dependencies. 
In the implementation, Docker could be used to containerize the application, ensuring its portability and ease of deployment across various systems while managing dependencies effectively. 
It can also aid in scaling and managing multiple instances of the application.



for the sake of being honest, I used chatGPT for help in this task



