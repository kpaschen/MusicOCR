# docker build -t opencv .
# docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v .:/developer/experimental opencv


FROM ubuntu:18.04

ARG OPENCV_VERSION="4.2.0"

RUN apt-get update 
RUN apt-get install -y curl build-essential checkinstall cmake
RUN apt-get install -y libleptonica-dev libtesseract-dev libgtk-3-dev
RUN apt-get install -y libgtest-dev

WORKDIR /development

RUN curl -sL https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz | tar xvz -C /development
RUN curl -sL https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.tar.gz | tar xvz -C /development

RUN cd /development/opencv-${OPENCV_VERSION} && mkdir build && cd build
RUN cmake -DOPENCV_EXTRA_MODULES_PATH=/development/opencv_contrib-${OPENCV_VERSION}/modules ..
RUN make

CMD /bin/bash


