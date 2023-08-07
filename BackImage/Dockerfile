FROM public.ecr.aws/lambda/python:3.10

# Install the function's dependencies using file requirements.txt
# from your project folder.
RUN mkdir -p ${LAMBDA_TASK_ROOT}/Sources
COPY . .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

RUN yum update -y && yum -y install mesa-libGL tar gzip zlib freetype-devel \
                                        gcc \
                                        ghostscript \
                                        lcms2-devel \
                                        libffi-devel \
                                        libimagequant-devel \
                                        libjpeg-devel \
                                        libraqm-devel \
                                        libtiff-devel \
                                        libwebp-devel \
                                        make \
                                        openjpeg2-devel \
                                        rh-python36 \
                                        rh-python36-python-virtualenv \
                                        sudo \
                                        tcl-devel \
                                        tk-devel \
                                        tkinter \
                                        which \
                                        xorg-x11-server-Xvfb \
                                        zlib-devel \
                                        makecache unzip \
                                         && yum clean all

RUN unzip ./tesseract-layer5.1.0best.zip -d /opt
# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD ["Sources.handler"]