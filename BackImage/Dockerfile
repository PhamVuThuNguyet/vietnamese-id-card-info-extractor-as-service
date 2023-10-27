FROM public.ecr.aws/lambda/python:3.10

# Install the function's dependencies using file requirements.txt
# from your project folder.
RUN mkdir -p ${LAMBDA_TASK_ROOT}/Sources
COPY . .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
RUN yum update -y && yum -y install wget
RUN mkdir -p ${LAMBDA_TASK_ROOT}/qrdet
RUN mkdir -p ${LAMBDA_TASK_ROOT}/qrdet/.model
ADD https://github.com/Eric-Canas/qrdet/releases/download/v2.0_release/qrdet-l.pt ${LAMBDA_TASK_ROOT}/qrdet/.model/
ADD current_release.txt ${LAMBDA_TASK_ROOT}/qrdet/.model/
RUN chmod -R 777 ${LAMBDA_TASK_ROOT}/qrdet/
RUN yum update -y && yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm &&\
sed -i "s/metalink=https/metalink=http/" /etc/yum.repos.d/epel.repo &&\
yum makecache &&\
yum -y install mesa-libGL zbar && yum clean all

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD ["Sources.handler"]