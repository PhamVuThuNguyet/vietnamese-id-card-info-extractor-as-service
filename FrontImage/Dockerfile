FROM public.ecr.aws/lambda/python:3.10

# Install the function's dependencies using file requirements.txt
# from your project folder.
RUN mkdir -p ${LAMBDA_TASK_ROOT}/Sources
COPY . .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

RUN yum update -y && yum -y install mesa-libGL && yum clean all

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD ["Sources.handler"]